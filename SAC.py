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
import torch.optim.lr_scheduler as lr_scheduler


class RunningMeanStd:
    def __init__(self, shape, mean=None, var=None, count=None):
        if mean is not None:
            self.mean = mean
            self.var = var
            self.count = count
        else:
            self.mean = np.zeros(shape, 'float64')
            self.var = np.ones(shape, 'float64')
            self.count = 1e-4 

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta ** 2 * self.count * batch_count / tot_count

        new_var = m2 / tot_count
        self.mean, self.var, self.count = new_mean, new_var, tot_count

class QuadcopterEnv(gym.Env):
    def __init__(self):
        super(QuadcopterEnv, self).__init__()
        self.action_space = gym.spaces.Box(low=0.0, high=1.0, shape=(4,), dtype=np.float32) 
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32)
        
        self.mass = 0.033
        self.inertia = np.diag(np.array([16.57e-6, 16.66e-6, 29.26e-6]))
        self.gravity = np.array([0, 0, -9.81*self.mass])  
        self.L = 0.028 
        self.Cb = 0.0059  
        self.Cd = 9.18e-7  
        self.dt = 0.05 
        self.max_time_step = 500
        self.episode = 0
        self.reset()

    def reset(self):
        self.episode += 1
        self.target_position = np.array([0.0, 0.0, 1.7])
        scaling_factor = min(1, self.episode/2000 + 0.2)

        x_range = np.array([-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5])
        y_range = np.array([-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5])
        z_range = np.array([1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2])

        self.position = self.target_position + scaling_factor * np.array([
            np.random.choice(x_range) - self.target_position[0],
            np.random.choice(y_range) - self.target_position[1],
            np.random.choice(z_range) - self.target_position[2]
        ])

        self.orientation = np.radians(np.random.choice(
            [-30.0, -25.0, -20.0, -15.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0], size=3))
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
        self.action = np.clip(action, self.action_space.low, self.action_space.high)
        action = self.action
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
        distance_reward = np.exp(-2.0 * distance)
        reward = 10 * distance_reward
        if distance < 0.01: 
            reward += 10
        return reward

    def _check_done(self, state):
        if np.linalg.norm(self.position - self.target_position) > 6.0 or self.time_step > self.max_time_step:
            return True
        return False

class CriticNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)  
        self.q_value = nn.Linear(hidden_dim, 1)
        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x)) 
        q_value = self.q_value(x)
        return q_value

class ActorNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, output_dim)
        self.log_std = nn.Linear(hidden_dim, output_dim)
        self.apply(self.init_weights)
        nn.init.constant_(self.log_std.bias, -0.5)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        mean = self.mean(x)
        log_std = torch.clamp(self.log_std(x), min=-20, max=2)
        std = log_std.exp()
        return mean, std

class SACAgent:
    def __init__(self, state_dim, action_dim, actor_lr=1e-4, critic_lr=3e-4, alpha_lr=1e-4, gamma=0.99, tau=0.005):
        self.state_dim = state_dim
        self.state_rms = RunningMeanStd(shape=(state_dim,), mean=None, var=None, count=None)

        self.actor = ActorNetwork(state_dim, action_dim)
        self.critic1 = CriticNetwork(state_dim + action_dim)
        self.critic2 = CriticNetwork(state_dim + action_dim)
        self.target_critic1 = CriticNetwork(state_dim + action_dim)
        self.target_critic2 = CriticNetwork(state_dim + action_dim)
        
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=critic_lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=critic_lr)
        self.log_alpha = torch.tensor(np.log(0.3), dtype=torch.float32, requires_grad=True)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)
        self.alpha = self.log_alpha.exp()

        self.actor_scheduler = lr_scheduler.StepLR(self.actor_optimizer, step_size=1000, gamma=0.99)
        self.critic1_scheduler = lr_scheduler.StepLR(self.critic1_optimizer, step_size=1000, gamma=0.99)
        self.critic2_scheduler = lr_scheduler.StepLR(self.critic2_optimizer, step_size=1000, gamma=0.99)
        self.alpha_scheduler = lr_scheduler.StepLR(self.alpha_optimizer, step_size=1000, gamma=0.99)

        self.gamma = gamma
        self.tau = tau
        self.memory = deque(maxlen=100000)
        self.batch_size = 32  
        self.target_entropy = -action_dim
        self.update_step = 0

    def select_action(self, state):
        self.state_rms.update(state[None, :])
        normalized_state = (state - self.state_rms.mean) / (np.sqrt(self.state_rms.var) + 1e-8)
        
        state = torch.FloatTensor(np.array(normalized_state)).unsqueeze(0)
        mean, std = self.actor(state)
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = (y_t + 1) / 2  
        action = action.clamp(0, 1)
        return action.cpu().detach().numpy()[0]

    def update(self):
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)
        
        state_np = np.array(state_batch)
        next_state_np = np.array(next_state_batch)
        
        self.state_rms.update(state_np)
        
        normalized_state_np = (state_np - self.state_rms.mean) / (np.sqrt(self.state_rms.var) + 1e-8)
        normalized_next_state_np = (next_state_np - self.state_rms.mean) / (np.sqrt(self.state_rms.var) + 1e-8)
        
        state = torch.FloatTensor(normalized_state_np)
        next_state = torch.FloatTensor(normalized_next_state_np)
        action = torch.FloatTensor(np.array(action_batch))
        reward = torch.FloatTensor(reward_batch).unsqueeze(1)
        done = torch.FloatTensor(done_batch).unsqueeze(1)
        
        with torch.no_grad():
            next_action_mean, next_action_std = self.actor(next_state)
            normal = torch.distributions.Normal(next_action_mean, next_action_std)
            x_t = normal.rsample()
            y_t = torch.tanh(x_t)
            next_action = (y_t + 1) / 2 
            log_prob = normal.log_prob(x_t) - torch.log(1 - y_t.pow(2) + 1e-6)
            log_prob = log_prob.sum(dim=-1, keepdim=True)
            target_q1 = self.target_critic1(next_state, next_action)
            target_q2 = self.target_critic2(next_state, next_action)
            target_q = torch.min(target_q1, target_q2) - self.alpha * log_prob
            target_q = reward + (1 - done) * self.gamma * target_q
        
        current_q1 = self.critic1(state, action)
        current_q2 = self.critic2(state, action)
        critic1_loss = F.mse_loss(current_q1, target_q)
        critic2_loss = F.mse_loss(current_q2, target_q)
        
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), max_norm=1.0)
        self.critic1_optimizer.step()
        
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), max_norm=1.0)
        self.critic2_optimizer.step()
        
        action_mean, action_std = self.actor(state)
        normal = torch.distributions.Normal(action_mean, action_std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = (y_t + 1) / 2  # Scale to [0, 1]
        log_prob = normal.log_prob(x_t) - torch.log(1 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        q1 = self.critic1(state, action)
        q2 = self.critic2(state, action)
        actor_loss = (self.alpha * log_prob - torch.min(q1, q2)).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        alpha_loss = -(self.log_alpha.exp() * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.alpha_optimizer.step()
        self.alpha = torch.clamp(self.log_alpha.exp(), 0, 0.2)
        
        self.actor_scheduler.step()
        self.critic1_scheduler.step()
        self.critic2_scheduler.step()
        self.alpha_scheduler.step()

        for target_param, param in zip(self.target_critic1.parameters(), self.critic1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for target_param, param in zip(self.target_critic2.parameters(), self.critic2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        self.update_step += 1

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
            'episode': episode,
            'norm_mean': self.state_rms.mean,
            'norm_var': self.state_rms.var,
            'norm_count': self.state_rms.count,
        }
        torch.save(checkpoint, filepath)

    def load_checkpoint(self, agent, filepath):
        checkpoint = torch.load(filepath)
        
        agent.actor.load_state_dict(checkpoint['actor_state_dict'])
        agent.critic1.load_state_dict(checkpoint['critic1_state_dict'])
        agent.critic2.load_state_dict(checkpoint['critic2_state_dict'])
        agent.target_critic1.load_state_dict(checkpoint['target_critic1_state_dict'])
        agent.target_critic2.load_state_dict(checkpoint['target_critic2_state_dict'])
        
        agent.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        agent.critic1_optimizer.load_state_dict(checkpoint['critic1_optimizer_state_dict'])
        agent.critic2_optimizer.load_state_dict(checkpoint['critic2_optimizer_state_dict'])
        agent.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])
        
        agent.log_alpha = checkpoint['log_alpha']
        agent.alpha = agent.log_alpha.exp()
        loaded_memory = deque(checkpoint['replay_buffer'], maxlen=100000)
        agent.memory = deque(loaded_memory, maxlen=100000) 
        
        all_rewards = checkpoint['all_rewards']
        all_states = checkpoint['all_states']
        episode = checkpoint['episode']

        self.state_rms = RunningMeanStd(shape=(self.state_dim,), mean=checkpoint['norm_mean'], var=checkpoint['norm_var'], count=checkpoint['norm_count'])

        return all_rewards, all_states, episode

env = QuadcopterEnv()
agent = SACAgent(state_dim=env.observation_space.shape[0], action_dim=env.action_space.shape[0])
num_episodes = 1000000
all_states = []
all_rewards, all_states, episode = agent.load_checkpoint(agent, "best_sac_checkpoint_with_5mm_reward.pth")

moving_avg = []
all_rewards = []
biggest_reward = 0
for episode in range(num_episodes):
    state = env.reset()
    pos = state[:3]
    angle = state[6:9]
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
    all_states.append(np.array(trajectory))
    all_rewards.append(total_reward)
    if episode > 50:
        moving_avg.append(np.mean(all_rewards[episode - 49 : episode + 1]))
        if moving_avg[-1] > biggest_reward:
            biggest_reward = moving_avg[-1]
            agent.save_checkpoint(agent, agent.memory, all_rewards, all_states, episode, "best_sac_checkpoint_with_1mm_reward.pth")
        print(f"Episode {episode + 1}, Total Reward: {total_reward:.2f}, Steps: {steps - 1} ({moving_avg[-1]:.2f})")
    else:
        print(f"Episode {episode + 1}, Total Reward: {total_reward:.2f}, Steps: {steps - 1} ({np.mean(all_rewards):.2f})")

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
