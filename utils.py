import gym
import torch
import cv2
import random
import os
import numpy as np
import matplotlib.pyplot as plt
from deepq_network import CNN, LinearMapNet


def discretize_state(state):
    """
    Convert continuous environment state into a discretized representation.
    This is useful for tabular Q-learning approaches.

    Each state dimension is scaled and clipped to [-2, 2].
    Last two elements are assumed to be already discrete.
    """
    discrete_state = (
        min(2, max(-2, int(state[0] / 0.05))),
        min(2, max(-2, int(state[1] / 0.1))),
        min(2, max(-2, int(state[2] / 0.1))),
        min(2, max(-2, int(state[3] / 0.1))),
        min(2, max(-2, int(state[4] / 0.1))),
        min(2, max(-2, int(state[5] / 0.1))),
        int(state[6]),
        int(state[7])
    )

    return discrete_state


def epsilon_greedy(q_func, state, eps, env_actions):
    """
    Select action using epsilon-greedy policy.

    With probability eps → explore (random action)
    Otherwise → exploit (choose best action from Q-function)
    """
    prob = np.random.random()

    # Exploration
    if prob < eps:
        return random.choice(range(env_actions))

    # Exploitation (Neural Network case)
    elif isinstance(q_func, CNN) or isinstance(q_func, LinearMapNet):
        with torch.no_grad():  # Disable gradient tracking (inference mode)
            return q_func(state).max(1)[1].item()

    # Exploitation (Tabular case)
    else:
        qvals = [q_func[state + (action,)] for action in range(env_actions)]
        return np.argmax(qvals)


def greedy(qstates_dict, state, env_actions):
    """
    Pure greedy policy (no exploration).
    Returns the maximum Q-value for a given state.
    """
    qvals = [qstates_dict[state + (action,)] for action in range(env_actions)]
    return max(qvals)


def discounted_return(episode_return, gamma):
    """
    Compute discounted return:
    G = r0 + gamma*r1 + gamma^2*r2 + ...
    """
    g = 0
    for i, r in enumerate(episode_return):
        g += (gamma ** i) * r

    return g


def decay_epsilon(curr_eps, exploration_final_eps):
    """
    Gradually decay epsilon (exploration rate).
    Stops decaying once minimum threshold is reached.
    """
    if curr_eps < exploration_final_eps:
        return curr_eps

    return curr_eps * 0.996


def get_frame(env):
    """
    Preprocess environment frame for CNN input:
    - Convert to grayscale
    - Resize to 84x84
    - Normalize pixel values
    - Convert to PyTorch tensor
    """
    # Get RGB frame from environment
    screen = env.render(mode='rgb_array')

    # Convert to grayscale
    frame = cv2.cvtColor(screen, cv2.COLOR_RGB2GRAY)

    # Resize for standard input size
    frame = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA)

    # Add channel dimension → (H, W, 1)
    frame = np.expand_dims(frame, -1)

    # Convert to (C, H, W) format required by PyTorch
    frame = frame.transpose((2, 0, 1))  # (1, 84, 84)

    # Normalize pixel values and convert to float tensor
    frame = np.ascontiguousarray(frame, dtype=np.float32) / 255.0
    frame = torch.from_numpy(frame)

    # Add batch dimension → (1, C, H, W)
    return frame.unsqueeze(0)


def lmn_input(obs):
    """
    Prepare input for LinearMapNet.
    Adds batch dimension and converts to tensor.
    """
    net_input = np.expand_dims(obs, 0)
    net_input = torch.from_numpy(net_input)

    return net_input


def build_qnetwork(env_actions, learning_rate, input_shape, network, device):
    """
    Initialize Q-network and optimizer.
    Supports:
    - CNN-based network
    - Linear mapping network
    """
    if network == 'cnn':
        qnet = CNN(env_actions)
    else:
        qnet = LinearMapNet(input_shape, env_actions)

    optimizer = torch.optim.RMSprop(qnet.parameters(), lr=learning_rate)

    return qnet.to(device), optimizer


def fit(qnet, qnet_optim, qtarget_net, loss_func,
        frames, actions, rewards, next_frames, dones,
        gamma, env_actions, device):
    """
    Perform one step of training using mini-batch:
    - Compute predicted Q-values
    - Compute TD targets
    - Calculate loss
    - Backpropagate and update weights
    """

    # -------- Current Q-values --------
    frames_t = torch.cat(frames).to(device)  # (batch, C, H, W)
    actions = torch.tensor(actions, device=device)

    q_t = qnet(frames_t)  # (batch, env_actions)

    # Select Q-values corresponding to taken actions
    q_t_selected = torch.sum(
        q_t * torch.nn.functional.one_hot(actions, env_actions), dim=1
    )

    # -------- Target Q-values --------
    dones = torch.tensor(dones, device=device)
    rewards = torch.tensor(rewards, device=device)

    frames_tp1 = torch.cat(next_frames).to(device)

    # Max Q-value for next state from target network
    q_tp1_best = qtarget_net(frames_tp1).max(1)[0].detach()

    # Zero out terminal states
    ones = torch.ones(dones.size(-1), device=device)
    q_tp1_best = (ones - dones) * q_tp1_best

    # TD target
    q_targets = rewards + gamma * q_tp1_best

    # -------- Loss & Optimization --------
    loss = loss_func(q_t_selected, q_targets)

    qnet_optim.zero_grad()
    loss.backward()
    qnet_optim.step()


def update_target_network(qnet, qtarget_net):
    """
    Copy weights from main Q-network to target network.
    This stabilizes training.
    """
    qtarget_net.load_state_dict(qnet.state_dict())


def save_model(qnet, episode, path):
    """
    Save model weights to disk.
    """
    torch.save(
        qnet.state_dict(),
        os.path.join(path, f'qnetwork_{episode}.pt')
    )


def plot_rewards(chosen_agents, agents_returns, num_episodes, window):
    """
    Plot moving average rewards for different agents.
    Helps visualize training performance.
    """
    num_intervals = int(num_episodes / window)

    for agent, agent_total_returns in zip(chosen_agents, agents_returns):
        print(len(agent_total_returns))
        print(f"\n{agent} lander average reward = {sum(agent_total_returns) / num_episodes}")

        # Compute moving average over windows
        l = []
        for j in range(num_intervals):
            l.append(
                round(np.mean(agent_total_returns[j * window: (j + 1) * window]), 1)
            )

        plt.plot(range(0, num_episodes, window), l)

    plt.xlabel("Episodes")
    plt.ylabel(f"Reward per {window} episodes")
    plt.title("RL Lander(s)")
    plt.legend(chosen_agents, loc="lower right")
    plt.show()