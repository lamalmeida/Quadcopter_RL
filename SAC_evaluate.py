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
        # Define action and observation space
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32)
        
        self.mass = 0.033
        self.inertia = np.diag(np.array([16.57e-6, 16.66e-6, 29.26e-6]))
        self.gravity = np.array([0, 0, -9.81 * self.mass])
        self.L = 0.028
        self.Cb = 0.0059
        self.Cd = 9.18e-7
        self.dt = 0.05
        self.max_time_step = 200
        self.episode = 1

        self.state = None
        self.reset()

    def reset(self):
        self.episode += 1
        self.target_position = np.array([0.0, 0.0, 1.7])
        self.position = np.array([0.0, 0.0, 1.7])

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
        action = np.clip(action, self.action_space.low, self.action_space.high)
        thrusts = action

        total_thrust = np.sum(thrusts)
        tau_x = self.L * self.Cb * (thrusts[1] - thrusts[3])
        tau_y = self.L * self.Cb * (thrusts[0] - thrusts[2])
        tau_z = self.Cd * (thrusts[0] - thrusts[1] + thrusts[2] - thrusts[3])
        torque = np.array([tau_x, tau_y, tau_z])

        R = self.rotation_matrix(*self.orientation)
        thrust_body_frame = np.array([0, 0, total_thrust])
        thrust_world_frame = R @ thrust_body_frame

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
            self.angular_velocity
        ))

        reward = self._calculate_reward(next_state)
        done = self._check_done(next_state)
        self.state = next_state
        return next_state, reward, done, {}

    def _calculate_reward(self, state):
        distance = np.linalg.norm(self.position - self.target_position)
        orientation_penalty = np.sum(np.abs(self.orientation)) * 0.2
        stability_penalty = np.sum(np.abs(self.angular_velocity)) * 0.1
        velocity_penalty = np.sum(np.abs(self.linear_velocity)) * 0.1
        
        distance_reward = np.exp(-20.0 * distance**2)
        reward = 100 * distance_reward - orientation_penalty - stability_penalty - velocity_penalty
        return reward

    def _check_done(self, state):
        return False

class ActorNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, output_dim)
        self.log_std = nn.Linear(hidden_dim, output_dim)
        self.batch_norm1 = nn.LayerNorm(hidden_dim)
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

def evaluate_actor_on_path(actor_model, env, steps_per_second=20, path_length=3.0, path_duration=6.0):
    state = env.reset()
    positions = [env.position.copy()]
    target_positions = []

    num_steps = int(path_duration * steps_per_second)
    max_steps = num_steps 
    step_distance = path_length / num_steps
    actor_model.eval()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    def update(step):
        nonlocal state
        step = step % num_steps
        if step < num_steps:
            if step < num_steps // 4:
                env.target_position[0] += step_distance  
            elif step < num_steps // 2:
                env.target_position[1] += step_distance
            elif step < 3 * num_steps // 4:
                env.target_position[0] -= step_distance
            else:
                env.target_position[1] -= step_distance
            
            target_positions.append(env.target_position.copy())

            relative_position = env.position - env.target_position
            state = np.concatenate((
                relative_position,
                env.linear_velocity,
                env.orientation,
                env.angular_velocity
            ))

            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                mean, std = actor_model(state_tensor)
                normal = torch.distributions.Normal(mean, std)
                action_sample = normal.sample()
                action = action_sample.clamp(-1, 1).squeeze().numpy()

            next_state, reward, done, _ = env.step(action)
            positions.append(env.position.copy())
            state = next_state

            ax.clear()
            ax.plot([p[0] for p in positions], [p[1] for p in positions], [p[2] for p in positions], label='Quadcopter Path', color='b')
            ax.plot([t[0] for t in target_positions], [t[1] for t in target_positions], [t[2] for t in target_positions], label='Target Path', color='r', linestyle='--')
            ax.set_xlabel('X Position')
            ax.set_ylabel('Y Position')
            ax.set_zlabel('Z Position')
            ax.set_title('Quadcopter Trajectory')
            ax.legend()
    
    ani = animation.FuncAnimation(fig, update, frames=max_steps, repeat=False)
    ani.save('quadcopter_path.gif', writer='imagemagick', fps=steps_per_second)
    plt.show()

    # Plotting X, Y, Z vs Time
    positions = np.array(positions)
    target_positions = np.array(target_positions)
    min_length = min(len(positions), len(target_positions))
    positions = positions[:min_length]
    target_positions = target_positions[:min_length]
    time = np.linspace(0, path_duration, len(positions))

    fig, axs = plt.subplots(3, 1, figsize=(10, 8))
    axs[0].plot(time, positions[:, 0], label='Actual X Position', color='b')
    axs[0].plot(time, target_positions[:, 0], label='Target X Position', color='r', linestyle='--')
    axs[0].set_xlabel('Time (s)')
    axs[0].set_ylabel('X Position')
    axs[0].legend()

    axs[1].plot(time, positions[:, 1], label='Actual Y Position', color='b')
    axs[1].plot(time, target_positions[:, 1], label='Target Y Position', color='r', linestyle='--')
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('Y Position')
    axs[1].legend()

    axs[2].plot(time, positions[:, 2], label='Actual Z Position', color='b')
    axs[2].plot(time, target_positions[:, 2], label='Target Z Position', color='r', linestyle='--')
    axs[2].set_xlabel('Time (s)')
    axs[2].set_ylabel('Z Position')
    axs[2].legend()

    plt.tight_layout()
    plt.show()

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath, map_location=torch.device('cpu'))  # Adjust device as needed
    actor_model = ActorNetwork(input_dim=12, output_dim=4)
    actor_model.load_state_dict(checkpoint['actor_state_dict'])
    actor_model.eval()
    return actor_model

if __name__ == "__main__":
    actor_model = load_checkpoint('sac_checkpoint.pth')
    env = QuadcopterEnv()
    evaluate_actor_on_path(actor_model, env)
