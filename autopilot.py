import torch
import gym
import sys
import numpy as np
from deepq_network import LinearMapNet
from utils import epsilon_greedy


def main():
    """
    Runs a trained DQN agent on the LunarLander-v2 environment.

    Arguments (from command line):
        sys.argv[1] -> number of episodes to run
        sys.argv[2] -> path to trained model (.pt file)
    """

    # Parse command-line arguments
    _, episodes, model_path = sys.argv

    # Initialize OpenAI Gym environment
    env = gym.make('LunarLander-v2')

    # Select device (GPU if available, else CPU)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Initialize Q-network
    # Input: 8-dimensional state vector
    # Output: 4 Q-values (one for each action)
    qnet = LinearMapNet(8, 4).to(device)

    # Load trained weights
    qnet.load_state_dict(torch.load(model_path, map_location=device))

    # Set network to evaluation mode (important: disables dropout, etc.)
    qnet.eval()

    # Run episodes
    for episode in range(int(episodes)):
        episode_reward = 0

        # Reset environment → initial state
        curr_state, done = env.reset(), False

        # Convert state to tensor and add batch dimension
        curr_state = np.expand_dims(curr_state, axis=0)
        curr_state = torch.from_numpy(curr_state).float()

        while not done:
            # Render environment (visualize agent behavior)
            env.render()

            # Select action using epsilon-greedy policy
            # ε ≈ 0 → almost greedy (exploitation)
            # This approximates: a = argmax_a Q(s,a)
            action = epsilon_greedy(qnet, curr_state.to(device), 0.0001, 4)

            # Take action in environment
            next_state, reward, done, _ = env.step(action)

            # Convert next state to tensor
            next_state = np.expand_dims(next_state, axis=0)
            next_state = torch.from_numpy(next_state).float()

            # Accumulate reward (episodic return)
            episode_reward += reward

            # Transition: s ← s'
            curr_state = next_state

        # Print total reward for this episode
        print(f"Episode {episode + 1}: Total Reward = {episode_reward:.2f}")

    # Close environment properly
    env.close()


if __name__ == '__main__':
    main()