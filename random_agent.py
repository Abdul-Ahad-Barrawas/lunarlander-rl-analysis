import gym
import sys
import numpy as np


def main():
    """
    Random Agent for LunarLander-v2

    This agent selects actions uniformly at random:
        a ~ Uniform(A)

    Purpose:
        - Serves as a baseline for comparison
        - Demonstrates environment difficulty
        - Helps quantify learning improvement of RL agents
    """

    # Parse number of episodes from command line
    episodes = int(sys.argv[-1])

    # Initialize environment
    env = gym.make('LunarLander-v2')

    for episode in range(episodes):
        episode_reward = 0

        # Reset environment → initial state
        state, done = env.reset(), False

        while not done:
            # Render environment (visual debugging / demonstration)
            env.render()

            # Random policy:
            # π(a|s) = 1 / |A|
            action = env.action_space.sample()

            # Take action in environment
            next_state, reward, done, _ = env.step(action)

            # Accumulate reward (episodic return)
            episode_reward += reward

            # Transition: s ← s'
            state = next_state

        # Print total reward for episode
        print(f"Episode {episode + 1}: Total Reward = {episode_reward:.2f}")

    # Clean shutdown
    env.close()


if __name__ == '__main__':
    main()