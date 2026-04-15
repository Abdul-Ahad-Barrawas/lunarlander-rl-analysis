import gym
import argparse
from rl_landers import *
import numpy as np


AGENTS_LIST = ["random", "monte-carlo", "sarsa", "q-learning", "dqn"]


def main():
    """
    Main Experiment Driver

    This script trains one or more RL agents on LunarLander-v2
    and compares their performance.

    It allows controlled experimentation over:
        - Number of episodes
        - Learning rate (α)
        - Discount factor (γ)
        - Exploration decay (ε)

    This enables reproducible benchmarking across algorithms.
    """

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--agents',
        nargs='+',
        help='Algorithms to train',
        choices=AGENTS_LIST,
        required=True
    )

    # Number of training episodes
    parser.add_argument(
        '--n_episodes',
        type=int,
        default=10000,
        help='Total training episodes'
    )

    # Learning rate α
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-3,
        help='Learning rate (α) for SARSA, Q-learning, DQN'
    )

    # Discount factor γ
    parser.add_argument(
        '--gamma',
        type=float,
        default=0.99,
        help='Discount factor (γ) controlling future reward importance'
    )

    # Final epsilon for ε-greedy
    parser.add_argument(
        '--final_eps',
        type=float,
        default=1e-2,
        help='Final exploration rate ε after decay'
    )

    args = parser.parse_args()

    # Initialize environment
    environment = gym.make("LunarLander-v2")

    chosen_agents = []
    agents_returns = []

    # Train selected agents
    for agent in args.agents:

        if agent == "random":
            print("\nTraining Random Agent...")
            total_rewards = random_lander(environment, args.n_episodes)

        elif agent == "monte-carlo":
            print(f"\nTraining Monte Carlo | Episodes={args.n_episodes}, γ={args.gamma}, ε_final={args.final_eps}")
            total_rewards = mc_lander(environment, args.n_episodes, args.gamma, args.final_eps)

        elif agent == "sarsa":
            print(f"\nTraining SARSA | Episodes={args.n_episodes}, α={args.lr}, γ={args.gamma}, ε_final={args.final_eps}")
            total_rewards = sarsa_lander(environment, args.n_episodes, args.gamma, args.lr, args.final_eps)

        elif agent == "q-learning":
            print(f"\nTraining Q-Learning | Episodes={args.n_episodes}, α={args.lr}, γ={args.gamma}, ε_final={args.final_eps}")
            total_rewards = qlearning_lander(environment, args.n_episodes, args.gamma, args.lr, args.final_eps)

        elif agent == "dqn":
            print(f"\nTraining DQN | Episodes={args.n_episodes}, α={args.lr}, γ={args.gamma}, ε_final={args.final_eps}")
            total_rewards = dqn_lander(environment, args.n_episodes, args.gamma, args.lr, args.final_eps)

        print("Done!")

        chosen_agents.append(agent)
        agents_returns.append(total_rewards)

    # Close environment
    environment.close()

    # Plot results (moving average window)
    win = 100
    plot_rewards(chosen_agents, agents_returns, args.n_episodes, win)


if __name__ == '__main__':
    main()