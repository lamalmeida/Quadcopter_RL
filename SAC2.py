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

device = torch.device("mps")

# === Environment Class for Quadcopter Control ===
class QuadcopterEnv(gym.Env):
    def __init__(self):
        super(QuadcopterEnv, self).__init__()
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32) 
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32)
        
        self.mass = 0.033
        self.inertia = np.diag(np.array([16.57e-6, 16.66e-6, 29.26e-6]))
        self.gravity = np.array([0, 0, -9.81*self.mass])  
        self.L = 0.028 
        self.Cb = 0.0059  
        self.Cd = 9.18e-7  
        self.dt = 0.05 
        self.max_time_step = 200
        self.episode = 0
        self.reset()

    def reset(self):
        self.episode += 1
        self.target_position = np.array([0.0, 0.0, 1.7])
        scaling_factor = min(1.0, self.episode / 2000 + 0.2)

        x_range = np.array([-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5])
        y_range = np.array([-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5])
        z_range = np.array([1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2])

        # Gradually scale the position to move from near the target to full random positions
        self.position = self.target_position + scaling_factor * np.array([
            np.random.choice(x_range) - self.target_position[0],
            np.random.choice(y_range) - self.target_position[1],
            np.random.choice(z_range) - self.target_position[2]
        ])

        self.orientation = np.radians(np.random.choice(
            [-44.69, -36.1, -26.93, -17.76, -9.17, 0.0, 9.17, 17.76, 26.93, 36.1, 44.69], size=3))
        self.linear_velocity = np.zeros(3)
        self.angular_velocity = np.zeros(3)

        relative_position = self.position - self.target_position
        self.state = np.concatenate((
            relative_position, 
            self.linear_velocity, 
            self.orientation, 
            self.angular_velocity
        ))
        self.time_step = 0
        return self.state

    def rotation_matrix(self, roll, pitch, yaw):
        c_r, s_r = np.cos(roll), np.sin(roll)
        c_p, s_p = np.cos(pitch), np.sin(pitch)
        c_y, s_y = np.cos(yaw), np.sin(yaw)

        return np.array([
            [c_y * c_p, c_y * s_p * s_r - s_y * c_r, c_y * s_p * c_r + s_y * s_r],
            [s_y * c_p, s_y * s_p * s_r + c_y * c_r, s_y * s_p * c_r - c_y * s_r],
            [-s_p, c_p * s_r, c_p * c_r]
        ])
    
    def step(self, action):
        # Simulate the quadcopter dynamics and update the state
        action = np.clip(action, self.action_space.low, self.action_space.high)
        total_thrust = np.sum(action)

        torque = np.array([
            self.L * self.Cb * (action[1] - action[3]),
            self.L * self.Cb * (action[0] - action[2]),
            self.Cd * (action[0] - action[1] + action[2] - action[3])
        ])

        R = self.rotation_matrix(*self.orientation)

        thrust_world_frame = R @ np.array([0, 0, total_thrust])

        total_force = thrust_world_frame + self.gravity
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

        reward = self._calculate_reward()
        done = self._check_done(next_state)
        self.state = next_state
        return next_state, reward, done, {}

    def _calculate_reward(self):
        distance = np.linalg.norm(self.position - self.target_position)
        orientation_penalty = np.sum(np.abs(self.orientation)) * 0.2
        stability_penalty = np.sum(np.abs(self.angular_velocity)) * 0.1
        velocity_penalty = np.sum(np.abs(self.linear_velocity)) * 0.1 
        
        # Smoother reward function
        distance_reward = np.exp(-20.0 * distance**2)  # Exponential decay
        reward = 100 * distance_reward - orientation_penalty - stability_penalty - velocity_penalty 
        return reward

    def _check_done(self, state):
        if np.linalg.norm(self.position - self.target_position) > 10.0 or self.time_step > self.max_time_step:
            return True
        return False

# === SAC Network Classes ===
class CriticNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
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
    def __init__(self, input_dim, output_dim, hidden_dim=64):
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
        mean = torch.sigmoid(self.mean(x))  
        log_std = torch.clamp(self.log_std(x), min=-20, max=20)
        std = log_std.exp()
        return mean, std

# === Improved SAC Agent ===
class SACAgent:
    def __init__(self, state_dim, action_dim, actor_lr=3e-4, critic_lr=1e-4, alpha_lr=1e-4, gamma=0.99, tau=0.005):
        # Initialize networks and move them to the available device
        self.device = device
        
        self.actor = ActorNetwork(state_dim, action_dim).to(self.device)
        self.critic1 = CriticNetwork(state_dim + action_dim).to(self.device)
        self.critic2 = CriticNetwork(state_dim + action_dim).to(self.device)
        self.target_critic1 = CriticNetwork(state_dim + action_dim).to(self.device)
        self.target_critic2 = CriticNetwork(state_dim + action_dim).to(self.device)
        
        # Copy weights from critic to target_critic
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=critic_lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=critic_lr)
        self.log_alpha = torch.tensor(np.log(0.2), dtype=torch.float32, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)
        self.alpha = self.log_alpha.exp()

        # Hyperparameters
        self.gamma = gamma
        self.tau = tau
        self.memory = deque(maxlen=1000000)
        self.batch_size = 4000  
        self.target_entropy = -action_dim
        self.i = 0

    def select_action(self, state):
        state = torch.FloatTensor(np.array(state)).unsqueeze(0).to(self.device)
        mean, std = self.actor(state)
        normal = torch.distributions.Normal(mean, std)
        action = normal.sample()
        action = action.clamp(-1, 1)  # Clamp action to valid range (-1, 1)
        return action.cpu().detach().numpy()[0]

    def update(self):
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        state, action, reward, next_state, done = zip(*batch)
        
        # Convert to torch tensors and move to device
        state = torch.FloatTensor(np.array(state)).to(self.device)
        action = torch.FloatTensor(np.array(action)).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        next_state = torch.FloatTensor(np.array(next_state)).to(self.device)
        done = torch.FloatTensor(done).unsqueeze(1).to(self.device)

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
        
    def save_checkpoint(self, agent, replay_buffer, all_rewards, all_states, episode, filepath):
        checkpoint = {
            'actor_state_dict': agent.actor.state_dict(),
            'critic1_state_dict': agent.critic1.state_dict(),
            'critic2_state_dict': agent.critic2.state_dict(),
            'target_critic1_state_dict': agent.target_critic1.state_dict(),
            'target_critic2_state_dict': agent.target_critic2.state_dict(),
            'actor_optimizer_state_dict': agent.actor_optimizer.state_dict(),
            'critic1_optimizer_state_dict': agent.critic1_optimizer.state_dict(),
            'critic2_optimizer_state_dict': agent.critic2_optimizer.state_dict(),
            'alpha_optimizer_state_dict': agent.alpha_optimizer.state_dict(),
            'log_alpha': agent.log_alpha,
            'replay_buffer': list(replay_buffer),
            'all_rewards': all_rewards,
            'all_states': all_states,
            'episode': episode
        }
        torch.save(checkpoint, filepath)

    def load_checkpoint(self, agent, filepath):
        checkpoint = torch.load(filepath, map_location=agent.device)
        
        # Load model states
        agent.actor.load_state_dict(checkpoint['actor_state_dict'])
        agent.critic1.load_state_dict(checkpoint['critic1_state_dict'])
        agent.critic2.load_state_dict(checkpoint['critic2_state_dict'])
        agent.target_critic1.load_state_dict(checkpoint['target_critic1_state_dict'])
        agent.target_critic2.load_state_dict(checkpoint['target_critic2_state_dict'])
        
        # Load optimizers
        agent.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        agent.critic1_optimizer.load_state_dict(checkpoint['critic1_optimizer_state_dict'])
        agent.critic2_optimizer.load_state_dict(checkpoint['critic2_optimizer_state_dict'])
        agent.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])
        
        # Load log alpha and replay buffer
        agent.log_alpha = checkpoint['log_alpha']
        agent.alpha = agent.log_alpha.exp()
        loaded_memory = deque(checkpoint['replay_buffer'], maxlen=1000000)
        agent.memory = deque(loaded_memory, maxlen=1000000) 
        
        # Load rewards, states, and episode count
        all_rewards = checkpoint['all_rewards']
        all_states = checkpoint['all_states']
        episode = checkpoint['episode']

        return all_rewards, all_states, episode

# === Training Loop ===
env = QuadcopterEnv()
agent = SACAgent(state_dim=env.observation_space.shape[0], action_dim=env.action_space.shape[0])
num_episodes = 120000
all_states = []
all_rewards = []
all_rewards, all_states, episode = agent.load_checkpoint(agent, "sac_checkpoint_prev.pth")

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
    if (episode + 1) % 100 == 0:
        agent.save_checkpoint(agent, agent.memory, all_rewards, all_states, episode, "sac_checkpoint.pth")
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

