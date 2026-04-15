import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Neural Network Architectures for Deep Q-Learning

We approximate the action-value function:

    Q(s, a; θ) ≈ expected return starting from state s, taking action a

This replaces the tabular Q(s,a) with a parameterised function.

Two architectures are provided:

1) CNN:
   Used for image-based environments (e.g., Atari from DeepMind Nature paper)
   Input: (batch_size, 1, 84, 84)

2) LinearMapNet:
   Used for low-dimensional state vectors (e.g., LunarLander: ℝ⁸)
"""


class CNN(nn.Module):
    """
    Convolutional Neural Network (DeepMind-style DQN)

    Architecture:
        Conv → ReLU → Conv → ReLU → Conv → ReLU → FC → ReLU → Output

    This network extracts spatial features from image input and maps them to Q-values.
    """

    def __init__(self, env_actions):
        super(CNN, self).__init__()

        # First convolutional layer:
        # Captures large spatial features (8x8 kernel, stride 4)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=8, stride=4)

        # Second convolutional layer:
        # Learns more detailed features (4x4 kernel, stride 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)

        # Third convolutional layer:
        # Refines features further (3x3 kernel)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Fully connected layer:
        # Flattened feature map → 512 hidden units
        # (64 × 7 × 7 = 3136)
        self.fc = nn.Linear(3136, 512)

        # Output layer:
        # Produces Q-values for each possible action
        self.out = nn.Linear(512, env_actions)

    def forward(self, x):
        """
        Forward pass computes Q(s, ·; θ)

        Input:
            x → state (image tensor)

        Output:
            Q-values for all actions
        """

        # Feature extraction via convolutions
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Flatten before fully connected layer
        x = self.conv_to_fc(x)

        # Nonlinear transformation
        x = F.relu(self.fc(x))

        # Final Q-values (no activation → regression output)
        return self.out(x)

    def conv_to_fc(self, x):
        """
        Flattens convolutional output into vector form.

        Converts:
            (batch_size, channels, height, width)
        → (batch_size, features)
        """
        size = x.size()[1:]  # ignore batch dimension

        num_features = 1
        for s in size:
            num_features *= s

        return x.view(-1, num_features)


class LinearMapNet(nn.Module):
    """
    Fully Connected Network for low-dimensional state spaces.

    Used for environments like LunarLander where:
        s ∈ ℝ⁸ (position, velocity, angle, etc.)

    Architecture:
        s → FC(64) → tanh → FC(64) → tanh → FC(actions)

    This approximates:
        Q(s,a; θ)
    """

    def __init__(self, input_shape, env_actions):
        super(LinearMapNet, self).__init__()

        # First hidden layer
        self.fc1 = nn.Linear(input_shape, 64)

        # Second hidden layer
        self.fc2 = nn.Linear(64, 64)

        # Output layer → Q-values for each action
        self.out = nn.Linear(64, env_actions)

    def forward(self, x):
        """
        Forward pass:

        Computes Q(s, a; θ) for all actions.

        Note:
        - tanh is used instead of ReLU → smoother gradients
        - No activation in output → Q-values are unbounded
        """

        # Hidden representation
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))

        # Output Q-values
        return self.out(x)