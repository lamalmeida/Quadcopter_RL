import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gym
import random
from collections import deque
import matplotlib.pyplot as plt
import pickle

# === Environment Class for Quadcopter Control ===
class QuadcopterEnv(gym.Env):
    def __init__(self):
        super(QuadcopterEnv, self).__init__()
        # Define action and observation space
        # Actions: thrust adjustments for each motor (continuous range)
        self.action_space = gym.spaces.Box(low=0.0, high=1.0, shape=(4,), dtype=np.float32)  # thrust for 4 motors
        
        # Observations: position, velocity, orientation, angular velocity
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32)
        
        self.mass = 0.033  
        self.inertia = np.diag(np.array([16.57e-6, 16.66e-6, 29.26e-6]))
        self.gravity = np.array([0, 0, -9.81])  
        self.L = 0.028 
        self.Cb = 0.0059  
        self.Cd = 9.18e-7  
        self.dt = 0.05 
        self.max_time_step = 200
        self.episode = 1

        # Initial state
        self.state = None
        self.reset()

    def reset(self):
        self.episode += 1
        # Reset the quadcopter state to a random initial state
        max_angle = np.pi 

        # Sample angle and azimuth
        angle = np.random.uniform(0, max_angle)  # Elevation angle from 0 to max_angle
        azimuth = np.random.uniform(0, 2 * np.pi)  # Azimuthal angle from 0 to 2*pi

        # Sample distance between inner and outer radius
        distance = -1

        # Convert spherical coordinates to Cartesian coordinates relative to the target position
        x_offset = distance * np.sin(angle) * np.cos(azimuth)
        y_offset = distance * np.sin(angle) * np.sin(azimuth)
        z_offset = distance * np.cos(angle)

        # Calculate final position relative to the target
        self.target_position = np.array([0.0, 0.0, 0.0])
        self.position = self.target_position + np.array([x_offset, y_offset, z_offset])
        relative_position = self.position - self.target_position
        
        # Random initial orientation from the specified set of values
        yaw = np.random.uniform(-np.pi, np.pi)
        roll_pitch_range = min(max(((self.episode) / 6000) * (np.pi), 0), np.pi/2)
        
        roll = np.random.uniform(-roll_pitch_range, roll_pitch_range)
        pitch = np.random.uniform(-roll_pitch_range, roll_pitch_range)
        
        self.orientation = np.array([roll, pitch, yaw])
        
        self.linear_velocity = np.zeros(3)
        self.angular_velocity = np.zeros(3)

        self.state = np.concatenate((
            relative_position, 
            self.linear_velocity, 
            self.orientation, 
            self.angular_velocity, 
        ))
        self.time_step = 0
        return self.state

    def rotation_matrix(self, roll, pitch, yaw):
        R_x = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)]
        ])

        R_y = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])

        R_z = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])

        R = R_z @ R_y @ R_x
        return R
    
    def step(self, action):
        # Simulate the quadcopter dynamics and update the state
        action = np.clip(action, self.action_space.low, self.action_space.high)
        thrusts = action  # Keep the action output as thrust for simplicity
        
        total_thrust = np.sum(thrusts)

        tau_x = self.L * self.Cb * (thrusts[1] - thrusts[3])
        tau_y = self.L * self.Cb * (thrusts[0] - thrusts[2])
        tau_z = self.Cd * (thrusts[0] - thrusts[1] + thrusts[2] - thrusts[3])
        torque = np.array([tau_x, tau_y, tau_z])

        R = self.rotation_matrix(*self.orientation)

        thrust_body_frame = np.array([0, 0, total_thrust])
        thrust_world_frame = R @ thrust_body_frame

        total_force = thrust_world_frame + self.gravity * self.mass
        self.linear_acceleration = total_force / self.mass 

        omega_cross_Jomega = np.cross(self.angular_velocity, self.inertia @ self.angular_velocity)
        self.angular_acceleration = np.linalg.inv(self.inertia) @ (torque - omega_cross_Jomega)

        self.linear_velocity += self.linear_acceleration * self.dt 
        self.position += self.linear_velocity * self.dt 

        self.angular_velocity += self.angular_acceleration * self.dt
        self.orientation += self.angular_velocity * self.dt

        self.orientation = (self.orientation + np.pi) % (2 * np.pi) - np.pi

        self.time_step += 1

        relative_position = self.position - self.target_position
        next_state = np.concatenate((
            relative_position, 
            self.linear_velocity, 
            self.orientation, 
            self.angular_velocity, 
        ))

        reward = self._calculate_reward(next_state)
        done = self._check_done(next_state)
        self.state = next_state
        return next_state, reward, done, {}

    def _calculate_reward(self, state):
        # Reward function for reaching and hovering at the target
        distance = np.linalg.norm(self.position - self.target_position)
        stability_penalty = np.abs(self.angular_velocity[0]) * 0.05 + np.abs(self.angular_velocity[1]) * 0.05 + np.abs(self.angular_velocity[2]) * 0.1 
        # smoothness_penalty = np.linalg.norm(self.linear_acceleration) * 0.1
        reward = 1.5 - distance - stability_penalty # - smoothness_penalty  # Negative reward proportional to distance, instability, and abrupt acceleration
        if distance < 0.1 + 0.9 * (2000 - self.episode)/2000:
            reward += 10  # Bonus for being close to the target
        return reward

    def _check_done(self, state):
        # Check if the quadcopter has reached a terminal condition
        if np.linalg.norm(self.position - self.target_position) > 6.0 or self.time_step > self.max_time_step:
            return True
        return False

# === SAC Network Classes ===
class CriticNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=512):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)  # Additional layer for more expressive power
        self.q_value = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))  # Additional layer
        q_value = self.q_value(x)
        return q_value

class ActorNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=512):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, output_dim)
        self.log_std = nn.Linear(hidden_dim, output_dim)
        self.batch_norm1 = nn.LayerNorm(hidden_dim)  # Batch Normalization for stability
        self.batch_norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, state):
        x = F.relu(self.batch_norm1(self.fc1(state)))
        x = F.relu(self.batch_norm2(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))
        mean = torch.tanh(self.mean(x))  # Use tanh to keep mean output in range (-1, 1)
        log_std = torch.clamp(self.log_std(x), min=-20, max=2)
        std = log_std.exp()
        return mean, std

# === Improved SAC Agent ===
class SACAgent:
    def __init__(self, state_dim, action_dim, actor_lr=3e-4, critic_lr=3e-4, alpha_lr=3e-4, gamma=0.99, tau=0.005):
        self.actor = ActorNetwork(state_dim, action_dim)
        self.critic1 = CriticNetwork(state_dim + action_dim)
        self.critic2 = CriticNetwork(state_dim + action_dim)
        self.target_critic1 = CriticNetwork(state_dim + action_dim)
        self.target_critic2 = CriticNetwork(state_dim + action_dim)
        
        # Copy weights from critic to target_critic
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=critic_lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=critic_lr)
        
        self.log_alpha = torch.tensor(np.log(0.2), requires_grad=True)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)
        self.alpha = self.log_alpha.exp()
        
        self.gamma = gamma
        self.tau = tau
        self.memory = deque(maxlen=1000000)
        self.batch_size = 2048  # Increased batch size for more stable learning
        self.target_entropy = -action_dim
        self.weight_decay = 0.0

    def select_action(self, state):
        state = torch.FloatTensor(np.array(state)).unsqueeze(0)  # Convert state to a numpy array before converting to tensor
        mean, std = self.actor(state)
        normal = torch.distributions.Normal(mean, std)
        action = normal.sample()
        action = action.clamp(-1, 1)  # Clamp action to valid range (-1, 1)
        return action.detach().numpy()[0]

    def update(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        state, action, reward, next_state, done = zip(*batch)
        
        # Convert to torch tensors
        state = torch.FloatTensor(np.array(state))
        action = torch.FloatTensor(np.array(action))  # Convert list of arrays to a single numpy array before converting to tensor
        reward = torch.FloatTensor(reward).unsqueeze(1)
        next_state = torch.FloatTensor(np.array(next_state))
        done = torch.FloatTensor(done).unsqueeze(1)
        
        # Compute target Q value
        with torch.no_grad():
            next_action_mean, next_action_std = self.actor(next_state)
            next_action_dist = torch.distributions.Normal(next_action_mean, next_action_std)
            next_action = next_action_dist.rsample()
            next_action_log_prob = next_action_dist.log_prob(next_action).sum(dim=-1, keepdim=True)
            next_action = next_action.clamp(-1, 1)
            target_q1 = self.target_critic1(next_state, next_action)
            target_q2 = self.target_critic2(next_state, next_action)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_action_log_prob
            target_q = reward + (1 - done) * self.gamma * target_q
        
        # Update critics
        current_q1 = self.critic1(state, action)
        current_q2 = self.critic2(state, action)
        critic1_loss = F.mse_loss(current_q1, target_q)
        critic2_loss = F.mse_loss(current_q2, target_q)
        
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()
        
        # Update actor
        action_mean, action_std = self.actor(state)
        action_dist = torch.distributions.Normal(action_mean, action_std)
        action_sample = action_dist.rsample()
        action_log_prob = action_dist.log_prob(action_sample).sum(dim=-1, keepdim=True)
        q1 = self.critic1(state, action_sample)
        q2 = self.critic2(state, action_sample)
        actor_loss = (self.alpha * action_log_prob - torch.min(q1, q2)).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Update alpha
        alpha_loss = -(self.log_alpha.exp() * (action_log_prob + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp()
        
        # Soft update target networks
        for target_param, param in zip(self.target_critic1.parameters(), self.critic1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for target_param, param in zip(self.target_critic2.parameters(), self.critic2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def save_model(self, model, filepath):
        """Save the model to the specified filepath."""
        torch.save(model.state_dict(), filepath)

    def load_model(self, model, filepath):
        """Load the model from the specified filepath."""
        model.load_state_dict(torch.load(filepath))
        model.eval()

# === Training Loop ===
env = QuadcopterEnv()
agent = SACAgent(state_dim=env.observation_space.shape[0], action_dim=env.action_space.shape[0])
num_episodes = 12000
all_states = []
all_rewards = []
# agent.load_model(agent.actor, "actor_model.pth")
# agent.load_model(agent.critic1, "critic1_model.pth")
# agent.load_model(agent.critic2, "critic2_model.pth")

for episode in range(num_episodes):
    state = env.reset()
    total_reward = 0
    done = False
    trajectory = []
    steps = 0
    while not done:
        trajectory.append(state.copy())
        action = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        agent.store_transition(state, action, reward, next_state, done)
        agent.update()
        state = next_state
        steps += 1
        total_reward += reward
    all_rewards.append(total_reward)
    all_states.append(np.array(trajectory))
    print(f"Episode {episode + 1}, Total Reward: {total_reward:.2f}, Steps: {steps - 1}")
    if episode % 1500 == 0:
        agent.save_model(agent.actor, "actor_model.pth")
        agent.save_model(agent.critic1, "critic1_model.pth")
        agent.save_model(agent.critic2, "critic2_model.pth")
        data_to_save = {
            "states": all_states,
            "rewards": all_rewards
        }
        with open("states_and_rewards.pkl", "wb") as f:
            pickle.dump(data_to_save, f)

plt.figure()
plt.plot(all_rewards)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Episode Rewards Over Time')
plt.show()

data_to_save = {
    "states": all_states,
    "rewards": all_rewards
}

with open("states_and_rewards.pkl", "wb") as f:
    pickle.dump(data_to_save, f)

agent.save_model(agent.actor, "actor_model.pth")
agent.save_model(agent.critic1, "critic1_model.pth")
agent.save_model(agent.critic2, "critic2_model.pth")

