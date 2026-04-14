# LunarLander-v2 — Reinforcement Learning Benchmark

Solving the [LunarLander-v2](https://gym.openai.com/envs/LunarLander-v2/) OpenAI Gym environment using four classical and deep RL algorithms. The environment provides an 8-dimensional continuous state space and 4 discrete actions (do nothing, fire left engine, fire main engine, fire right engine). An episode is solved when the agent accumulates ≥ 200 reward.

---

## Table of Contents
1. [Background](#background)
2. [Algorithms & Mathematics](#algorithms--mathematics)
   - [Monte-Carlo](#1-every-visit-monte-carlo)
   - [Sarsa](#2-sarsa-on-policy-td0)
   - [Q-learning](#3-q-learning-off-policy-td0)
   - [DQN](#4-deep-q-network-dqn)
3. [Training Clips](#training-clips)
4. [Experiments](#experiments)
5. [Execution](#execution)
6. [Implementation References](#implementation-references)

---

## Background

All algorithms here operate within the standard **Markov Decision Process (MDP)** framework, defined by the tuple $(\mathcal{S}, \mathcal{A}, P, R, \gamma)$:

| Symbol | Meaning |
|--------|---------|
| $\mathcal{S}$ | State space (8-dim continuous vector) |
| $\mathcal{A}$ | Action space $\{0, 1, 2, 3\}$ |
| $P(s'\mid s,a)$ | Transition probability |
| $R(s,a)$ | Reward function |
| $\gamma \in [0,1)$ | Discount factor |

The **action-value function** under policy $\pi$ is:

$$Q^{\pi}(s, a) = \mathbb{E}_{\pi}\left[\sum_{k=0}^{\infty} \gamma^k R_{t+k+1} \;\mid\; S_t = s,\, A_t = a\right]$$

The optimal action-value function satisfies the **Bellman optimality equation**:

$$Q^{*}(s, a) = \mathbb{E}\left[R_{t+1} + \gamma \max_{a'} Q^{*}(S_{t+1}, a') \;\mid\; S_t = s,\, A_t = a\right]$$

All tabular methods discretise the continuous state space via **tile coding** before applying their update rules.

---

## Algorithms & Mathematics

### 1. Every-Visit Monte Carlo

Monte Carlo (MC) methods learn directly from complete episodes. No bootstrapping is used — updates wait until the episode terminates.

**Return:** The discounted cumulative reward from timestep $t$ is:

$$G_t = \sum_{k=0}^{T-t-1} \gamma^k R_{t+k+1}$$

**Policy Evaluation (Every-Visit):** For every visit to state-action pair $(s, a)$ in an episode, the running average is updated:

$$Q(s, a) \leftarrow Q(s, a) + \frac{1}{N(s,a)} \left[G_t - Q(s, a)\right]$$

where $N(s, a)$ is the total visit count. This is a **true running average** — it does not use a fixed learning rate $\alpha$. As $N(s,a) \to \infty$, $Q(s,a) \to Q^{\pi}(s,a)$ by the law of large numbers.

**Policy Improvement ($\varepsilon$-greedy):**

$$\pi(a \mid s) = \begin{cases} 1 - \varepsilon + \dfrac{\varepsilon}{\lvert\mathcal{A}\rvert} & \text{if}\; a = \arg\max_{a'} Q(s, a') \\ \dfrac{\varepsilon}{\lvert\mathcal{A}\rvert} & \text{otherwise} \end{cases}$$

> Because MC uses a running average rather than a fixed $\alpha$, the learning rate hyperparameter has **no effect** on its training curve, as confirmed in the experiments below.

---

### 2. Sarsa (On-Policy TD(0))

Sarsa is an **on-policy** Temporal Difference method. Unlike MC, it bootstraps — updating at every step using the very next state-action pair. The name encodes the full quintuple used in each update: $(S_t, A_t, R_{t+1}, S_{t+1}, A_{t+1})$.

**TD Error:**

$$\delta_t = R_{t+1} + \gamma\, Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t)$$

**Update Rule:**

$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha\, \delta_t$$

where $\alpha$ is the step-size (learning rate). Since $A_{t+1}$ is sampled from the current $\varepsilon$-greedy policy, Sarsa is on-policy: it evaluates the same policy it is improving. Sarsa converges to $Q^{*}$ under standard stochastic approximation conditions (GLIE schedule for $\varepsilon$, diminishing $\alpha$).

---

### 3. Q-learning (Off-Policy TD(0))

Q-learning is an **off-policy** TD method. The key difference from Sarsa is that the bootstrap target uses the greedy action at $S_{t+1}$, regardless of which action the behaviour policy actually takes.

**Update Rule:**

$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \left[ R_{t+1} + \gamma \max_{a'} Q(S_{t+1}, a') - Q(S_t, A_t) \right]$$

The target $R_{t+1} + \gamma \max_{a'} Q(S_{t+1}, a')$ is a direct approximation of the Bellman optimality operator applied to $Q$. Because it decouples the behaviour policy from the target policy, Q-learning directly estimates $Q^{*}$ without requiring the behaviour policy to be greedy. This makes it more aggressive and often faster-converging than Sarsa, but at the cost of higher variance.

**Sarsa vs. Q-learning — the key distinction:**

| | Sarsa | Q-learning |
|---|---|---|
| Policy type | On-policy | Off-policy |
| Bootstrap target | $Q(S_{t+1}, A_{t+1})$ | $\max_{a'} Q(S_{t+1}, a')$ |
| Converges to | $Q^{\pi_\varepsilon}$ | $Q^{*}$ |
| Sensitivity to $\alpha$ | High | High |

---

### 4. Deep Q-Network (DQN)

DQN replaces the tabular $Q$-table with a neural network $Q(s, a;\, \theta)$ parameterised by weights $\theta$, making it applicable to the raw continuous state space without tile coding.

**Network Architecture:**

$$Q(s,\cdot\,;\theta): \mathbb{R}^8 \xrightarrow{\text{FC + ReLU}} \mathbb{R}^{64} \xrightarrow{\text{FC + ReLU}} \mathbb{R}^{64} \xrightarrow{\text{FC}} \mathbb{R}^{4}$$

The network outputs $Q$-values for all four actions simultaneously in a single forward pass.

**Loss Function:**

$$\mathcal{L}(\theta) = \mathbb{E}_{(s,a,r,s') \sim \mathcal{D}}\left[\left(y - Q(s, a;\, \theta)\right)^2\right]$$

where the **TD target** $y$ is computed using a periodically frozen **target network** $\theta^-$:

$$y = r + \gamma \max_{a'} Q(s', a';\, \theta^-)$$

**Two stabilisation techniques** distinguish DQN from naïve neural Q-learning:

1. **Experience Replay.** Transitions $(s_t, a_t, r_{t+1}, s_{t+1})$ are stored in a replay buffer $\mathcal{D}$ and sampled i.i.d. at training time. This breaks the temporal correlations between consecutive transitions that would otherwise destabilise SGD.

2. **Target Network.** A separate copy $\theta^-$ (updated every $C$ steps via $\theta^- \leftarrow \theta$) provides stable regression targets. Without it, both the prediction and the target move simultaneously, causing the loss surface to shift under the optimizer.

**Gradient update** (Adam optimiser):

$$\theta \leftarrow \theta - \eta \nabla_{\theta} \mathcal{L}(\theta)$$

**$\varepsilon$-Greedy Exploration with Decay:**

$$\varepsilon_t = \max\left(\varepsilon_{\text{final}},\; \varepsilon_0 \cdot e^{-\lambda t}\right)$$

where $\varepsilon_0 = 1.0$ and $\varepsilon_{\text{final}} = 0.01$ by default. The exponential decay ensures broad exploration early and exploitation-dominant behaviour later.

> DQN requires a small learning rate ($\alpha \approx 0.001$) to prevent the nonlinear approximator from diverging. This is consistent with the experiment results — DQN achieves its best performance (206.948 avg. reward) at $\alpha = 0.001$ and performs worst at $\alpha = 0.1$.

---

## Training Clips

Snapshots captured near the end of training (~10,000 episodes). A random agent is provided for comparison.

### Random

<p align="center">
  <img src="data/random.gif" width="400"/>
</p>


### Monte-Carlo

<p align="center">
  <img src="data/monte_carlo.gif" width="400"/>
</p>

### Sarsa

<p align="center">
  <img src="data/sarsa.gif" width="400"/>
</p>

### Q-learning

<p align="center">
  <img src="data/qlearning.gif" width="400"/>
</p>

### DQN

<p align="center">
  <img src="data/dqn.gif" width="400"/>
</p>
---

## Experiments

Average reward over 10,000 training episodes across three learning rates.

1. `lr=0.1`
![all_agents_lr01](data/all_lr01.png)

2. `lr=0.01`
![all_agents_lr001](data/all_lr001.png)

3. `lr=0.001`
![all_agents_lr0001](data/all_lr0001.png)

Note that the learning rate only affects Sarsa, Q-learning and DQN. The Every-Visit Monte Carlo implementation uses a true running average in its policy evaluation step, so $\alpha$ has no effect on its training progress.

|             | lr = 0.1 | lr = 0.01 | lr = 0.001 |
| ----------- | --------- | ---------- | ----------- |
| Monte-Carlo | -23.134   | **-53.209** | -44.191 |
| Sarsa       | 95.445    | -55.731    | -182.449   |
| Q-learning  | **95.793** | -77.473   | -173.590   |
| DQN         | -150.694  | -70.869    | **206.948** |

**Key takeaways:**
- **Sarsa and Q-learning** converge fastest at `lr=0.1`. Higher learning rates allow rapid adaptation in the early tabular regime, though they risk instability in dense-reward regions.
- **DQN** benefits from a small learning rate (`lr=0.001`) that prevents the neural approximator from oscillating. It is the only method to exceed +200 avg. reward, crossing the environment's solve threshold.
- **Monte-Carlo** is learning-rate agnostic (running average) and shows moderate, stable performance across all conditions. Its high variance stems from the noisy return estimates over full episodes.

---

## Execution

To train all agents with default hyperparameters (`n_episodes=10000`, `lr=0.001`, `gamma=0.99`, `final_eps=0.01`):

```bash
python train.py --agents random monte-carlo sarsa q-learning dqn
```

For a customised run (e.g. DQN only):

```bash
python train.py --agents dqn --n_episodes 3000 --lr 0.0001 --gamma 0.99 --final_eps 0.02
```

To test a trained DQN checkpoint:

```bash
python autopilot.py <num_episodes> models/*/qnetwork_{*}.pt
```

To benchmark against a random baseline:

```bash
python random_agent.py <num_episodes>
```

> A pretrained checkpoint [`models/pretrained/qnetwork_2000.pt`](models/pretrained/qnetwork_2000.pt) lands the lunar lander optimally **95% of the time**.

---

## Implementation References

1. [OpenAI Baselines](https://github.com/openai/baselines)
2. [Reinforcement Learning (DQN) Tutorial — PyTorch](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)
3. [Solving The Lunar Lander Problem under Uncertainty using Reinforcement Learning](https://arxiv.org/abs/2011.11850)
4. Sutton, R. S. & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.