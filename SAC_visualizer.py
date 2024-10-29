import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import imageio
from io import BytesIO

# Load the saved data
with open("states_and_rewards.pkl", "rb") as f:
    data = pickle.load(f)

all_states = data["states"]
all_rewards = data["rewards"]

# Specify the episodes you want to plot
episodes = [1500, 3000, 6000]  # Change this to the list of episodes you want to plot

# Plot the trajectories and orientations for the specified episodes
for episode in episodes:
    episode -= 1
    if episode < len(all_states):
        states = all_states[episode]
        rewards = all_rewards[episode]

        # Extract position and orientation (roll, pitch, yaw) from states
        positions = states[:, :3]
        orientations = states[:, 6:9]

        x_coords = positions[:, 0]
        y_coords = positions[:, 1]
        z_coords = positions[:, 2]

        # Create 3D gif of trajectory with orientation
        fig = plt.figure(figsize=(8, 6))  # Specify a fixed figure size
        ax = fig.add_subplot(111, projection='3d')

        # Set fixed camera view
        ax.set_xlim([-6.5, 6.5])
        ax.set_ylim([-6.5, 6.5])
        ax.set_zlim([-6.5, 6.5])
        ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio for all axes
        ax.view_init(elev=20, azim=30)  # Set a default elevation and azimuth angle for the view

        gif_frames = []

        for i in range(len(positions)):
            ax.clear()
            ax.set_xlim([-6.5, 6.5])
            ax.set_ylim([-6.5, 6.5])
            ax.set_zlim([-6.5, 6.5])
            ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio for all axes
            ax.view_init(elev=20, azim=30)  # Keep the camera view consistent

            ax.plot(x_coords[:i+1], y_coords[:i+1], z_coords[:i+1], color='b', label=f'Episode {episode + 1}')
            
            # Orientation (roll, pitch, yaw) lines (3 separate lines with fixed length)
            length = 0.5  # Fixed length for all orientation lines
            ax.plot([positions[i, 0], positions[i, 0] + length * np.cos(orientations[i, 0])],
                    [positions[i, 1], positions[i, 1]],
                    [positions[i, 2], positions[i, 2]],
                    color='r', linewidth=2, label='Roll' if i == 0 else None)
            ax.plot([positions[i, 0], positions[i, 0]],
                    [positions[i, 1], positions[i, 1] + length * np.cos(orientations[i, 1])],
                    [positions[i, 2], positions[i, 2]],
                    color='g', linewidth=2, label='Pitch' if i == 0 else None)
            ax.plot([positions[i, 0], positions[i, 0]],
                    [positions[i, 1], positions[i, 1]],
                    [positions[i, 2], positions[i, 2] + length * np.cos(orientations[i, 2])],
                    color='b', linewidth=2, label='Yaw' if i == 0 else None)

            ax.set_title(f'Trajectory for Episode {episode + 1} (Reward: {rewards})')
            ax.set_xlabel('X Coordinate')
            ax.set_ylabel('Y Coordinate')
            ax.set_zlabel('Z Coordinate')
            ax.legend()
            plt.grid(True)

            # Save each frame for gif creation using BytesIO buffer
            buf = BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            image = imageio.imread(buf)
            gif_frames.append(image)

        # Save the gif
        imageio.mimsave(f'episode_{episode + 1}_trajectory.gif', gif_frames, fps=10)

        plt.show()
    else:
        print(f"Episode {episode + 1} not found.")