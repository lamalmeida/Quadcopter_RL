import torch
import gym
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.animation as animation

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
        self.max_time_step = 200
        self.episode = 0
        self.reset()

    def reset(self):
        self.episode += 1
        self.target_position = np.array([0.0, 0.0, 1.7])
        self.position = np.zeros(3)
        self.orientation = np.zeros(3)
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
            return False
        return False

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

def evaluate_actor_with_initial_orientation(actor_model, env, path_duration=5.0, use_relative_velocity=False, norm_mean=None, norm_var=None):
    state = env.reset()
    
    env.orientation = np.array([0, np.pi/4, 0])  
    env.position = np.array([0.0, 0.0, 2.0])
    env.linear_velocity = np.zeros(3)  
    env.angular_velocity = np.zeros(3) 

    positions = []
    target_positions = []

    num_steps = int(path_duration * 20)
    actor_model.eval()

    for step in range(num_steps):
        env.target_position = np.array([0.0, 0.0, 0.0])
        target_positions.append(env.target_position.copy())

        relative_position = env.position - env.target_position

        velocity = env.linear_velocity

        state = np.concatenate((
            relative_position,
            velocity,
            env.orientation,
            env.angular_velocity
        ))

        state = (state - norm_mean) / (np.sqrt(norm_var))
        state_tensor = torch.FloatTensor(state)
        with torch.no_grad():
            mean, std = actor_model(state_tensor)
            normal = torch.distributions.Normal(mean, std)
            x_t = normal.rsample()
            y_t = torch.tanh(x_t)
            action = (y_t + 1) / 2  
            action = action.clamp(0, 1).numpy()

        next_state, reward, done, _ = env.step(action)
        positions.append(env.position.copy())
        state = next_state

    return np.array(positions), np.array(target_positions)

def evaluate_actor_on_straight_path(actor_model, env, path_length=5.0, path_duration=5, use_relative_velocity=False, norm_mean=None, norm_var=None):
    state = env.reset()
    env.position = np.array([0.0, 0.0, 1.7]) 
    positions = [env.position.copy()]
    target_positions = [env.position.copy()]

    num_steps = int(path_duration * 20)
    step_size = path_length / num_steps 
    actor_model.eval()

    for step in range(num_steps):
        env.target_position[0] += step_size  
        target_positions.append(env.target_position.copy())

        relative_position = env.position - env.target_position
        if use_relative_velocity:
            target_velocity = np.array([step_size * 20, 0, 0]) 
            relative_velocity = env.linear_velocity - target_velocity
            velocity = relative_velocity
        else:
            velocity = env.linear_velocity

        state = np.concatenate((
            relative_position,
            velocity,
            env.orientation,
            env.angular_velocity
        ))

        if norm_mean is not None and norm_var is not None:
            state = (state - norm_mean) / np.sqrt(norm_var)

        state_tensor = torch.FloatTensor(state)
        with torch.no_grad():
            mean, std = actor_model(state_tensor)
            normal = torch.distributions.Normal(mean, std)
            x_t = normal.rsample()
            y_t = torch.tanh(x_t)
            action = (y_t + 1) / 2 
            action = action.clamp(0, 1).numpy()

        next_state, reward, done, _ = env.step(action)
        positions.append(env.position.copy())
        state = next_state

    return np.array(positions), np.array(target_positions)

def evaluate_actor_on_path(actor_model, env, path_radius=1.0, path_duration=2*np.pi, use_relative_velocity=False, norm_mean=None, norm_var=None):
    state = env.reset()
    env.position = np.array([1.0, 0.0, 1.7])
    positions = [env.position.copy()]
    target_positions = [env.position.copy()]

    num_steps = int(path_duration * 20)
    actor_model.eval()

    angular_velocity = 2 * np.pi / path_duration 
    for step in range(num_steps):
        angle = angular_velocity * (step / num_steps) * path_duration

        env.target_position[0] = path_radius * np.cos(angle)
        env.target_position[1] = path_radius * np.sin(angle)
        target_positions.append(env.target_position.copy())

        relative_position = env.position - env.target_position
        if use_relative_velocity:
            target_velocity = np.array([
                -path_radius * angular_velocity * np.sin(angle),
                path_radius * angular_velocity * np.cos(angle),
                0
            ])
            relative_velocity = env.linear_velocity - target_velocity
            velocity = relative_velocity
        else:
            velocity = env.linear_velocity

        state = np.concatenate((
            relative_position,
            velocity,
            env.orientation,
            env.angular_velocity
        ))

        state = (state - norm_mean) / (np.sqrt(norm_var))
        state_tensor = torch.FloatTensor(state)
        with torch.no_grad():
            mean, std = actor_model(state_tensor)
            normal = torch.distributions.Normal(mean, std)
            x_t = normal.rsample()
            y_t = torch.tanh(x_t)
            action = (y_t + 1) / 2  
            action = action.clamp(0, 1).numpy()

        next_state, reward, done, _ = env.step(action)
        positions.append(env.position.copy())
        state = next_state

    return np.array(positions), np.array(target_positions)

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath, map_location=torch.device('cpu')) 
    actor_model = ActorNetwork(input_dim=12, output_dim=4)
    actor_model.load_state_dict(checkpoint['actor_state_dict'])
    actor_model.eval()
    norm_mean = checkpoint['norm_mean']
    norm_var = checkpoint['norm_var']
    return actor_model, norm_mean, norm_var

def evaluate_with_wind(actor_model, env, wind_force_magnitude=0.1, path_duration=5.0, norm_mean=None, norm_var=None):
    state = env.reset()
    env.position = np.array([1.0, 1.0, 0.0])  
    env.target_position = np.array([0.0, 0.0, 1.0]) 
    env.linear_velocity = np.zeros(3)
    env.angular_velocity = np.zeros(3)
    env.orientation = np.zeros(3) 

    positions = [env.position.copy()]
    target_positions = [env.target_position.copy()]
    wind_randomness = 0.4

    num_steps = int(path_duration * 20)
    actor_model.eval()

    for step in range(num_steps):
        wind_force = np.random.uniform(wind_force_magnitude-wind_randomness, wind_force_magnitude+wind_randomness, size=3)
        
        env.position += wind_force * env.dt

        relative_position = env.position - env.target_position
        vel = env.linear_velocity + np.array([wind_force_magnitude,wind_force_magnitude,wind_force_magnitude])
        state = np.concatenate((
            relative_position,
            vel,
            env.orientation,
            env.angular_velocity
        ))

        state = (state - norm_mean) / (np.sqrt(norm_var) + 1e-8)
        state_tensor = torch.FloatTensor(state)
        
        with torch.no_grad():
            mean, std = actor_model(state_tensor)
            normal = torch.distributions.Normal(mean, std)
            x_t = normal.rsample()
            y_t = torch.tanh(x_t)
            action = (y_t + 1) / 2  
            action = action.clamp(0, 1).numpy()

        next_state, reward, done, _ = env.step(action)
        positions.append(env.position.copy())
        target_positions.append(env.target_position.copy())
        state = next_state

    return np.array(positions), np.array(target_positions)

def experiment_immovable_target(env, actor_model, norm_mean, norm_var, duration=12.5, dt=0.05):
    num_steps = int(duration / dt)
    time = np.linspace(0, duration, num_steps)
    
    state = env.reset()
    env.position = np.array([0.5, 0.5, 1.8])  
    env.target_position = np.array([0.0, 0.0, 1.7]) 
    positions = []
    angular_velocities = []
    
    for step in range(num_steps):
        relative_position = env.position - env.target_position
        state = np.concatenate((
            relative_position,
            env.linear_velocity,
            env.orientation,
            env.angular_velocity
        ))

        state = (state - norm_mean) / (np.sqrt(norm_var) + 1e-8)
        state_tensor = torch.FloatTensor(state)
        with torch.no_grad():
            mean, std = actor_model(state_tensor)
            normal = torch.distributions.Normal(mean, std)
            x_t = normal.rsample()
            y_t = torch.tanh(x_t)
            action = (y_t + 1) / 2  
            action = action.clamp(0, 1).numpy()
        
        next_state, _, _, _ = env.step(action)
        positions.append(env.position.copy())
        angular_velocities.append(env.angular_velocity.copy())
        state = next_state
    
    return time, np.array(positions), np.array(angular_velocities)

def plot_results(global_positions, global_slow, target_positions, target_slow, ftime=6.3, stime=12.6):
    global_positions = np.array(global_positions)
    global_slow = np.array(global_slow)
    target_positions = np.array(target_positions)
    
    fast_steps = global_positions.shape[0]
    slow_steps = global_slow.shape[0]
    time_fast = np.linspace(0, ftime, fast_steps)
    time_slow = np.linspace(0, stime, slow_steps)
    
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(global_positions[:, 0], global_positions[:, 1], global_positions[:, 2], color='red', linestyle='-', label='Fast Positions')
    ax.plot(global_slow[:, 0], global_slow[:, 1], global_slow[:, 2], color='blue', linestyle='-', label='Slow Positions')
    ax.plot(target_positions[:, 0], target_positions[:, 1], target_positions[:, 2], color='red', linestyle='--', label='Target Fast Positions')
    ax.plot(target_slow[:, 0], target_slow[:, 1], target_slow[:, 2], color='blue', linestyle='--', label='Target Slow Positions')
    ax.set_title('3D Plot of Positions')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.show()

    fig, axes = plt.subplots(1, 3, figsize=(12, 6), sharex=True)
    
    axes[0].plot(time_fast, global_positions[:, 0], color='red', linestyle='-', label='Fast X')
    axes[0].plot(time_slow, global_slow[:, 0], color='blue', linestyle='-', label='Slow X')
    axes[0].plot(time_fast, target_positions[:, 0], color='red', linestyle='--', label='Fast Target X')
    axes[0].plot(time_slow, target_slow[:, 0], color='blue', linestyle='--', label='Slow Target X')
    axes[0].set_title('X-Direction')
    axes[0].set_ylabel('X Position')
    axes[0].legend()
    
    axes[1].plot(time_fast, global_positions[:, 1], color='red', linestyle='-', label='Fast Y')
    axes[1].plot(time_slow, global_slow[:, 1], color='blue', linestyle='-', label='Slow Y')
    axes[1].plot(time_fast, target_positions[:, 1], color='red', linestyle='--', label='Fast Target Y')
    axes[1].plot(time_slow, target_slow[:, 1], color='blue', linestyle='--', label='Slow Target Y')
    axes[1].set_title('Y-Direction')
    axes[1].set_ylabel('Y Position')
    axes[1].legend()
    
    axes[2].plot(time_fast, global_positions[:, 2], color='red', linestyle='-', label='Fast Z')
    axes[2].plot(time_slow, global_slow[:, 2], color='blue', linestyle='-', label='Slow Z')
    axes[2].plot(time_fast, target_positions[:, 2], color='red', linestyle='--', label='Fast Target Z')
    axes[2].plot(time_slow, target_slow[:, 2], color='blue', linestyle='--', label='Slow Target Z')
    axes[2].set_title('Z-Direction')
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('Z Position')
    axes[2].legend()
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    actor_model, norm_mean, norm_var = load_checkpoint('best_sac_checkpoint_with_1cm_reward.pth')
    env = QuadcopterEnv()

    global_positions, target_positions = evaluate_with_wind(actor_model, env, wind_force_magnitude=1, path_duration=10, norm_mean=norm_mean, norm_var=norm_var)

    global_slow, target_slow = evaluate_with_wind(actor_model, env, wind_force_magnitude=0.2, path_duration=10, norm_mean=norm_mean, norm_var=norm_var)

    plot_results(global_positions, global_slow, target_positions, target_slow, ftime=10, stime=10)

    
