# Tic-Tac-Toe AI with PPO
This project implements a reinforcement learning agent that learns to play Tic-Tac-Toe using the Proximal Policy Optimization (PPO) algorithm. The agent is trained to play against itself, and the goal is to maximize its chances of winning or drawing the game.

## How to run the code
1. Install Dependencies

Make sure the following Python packages installed:
- torch: For neural network implementation and tensor operations
- tqdm: For displaying progress bars during training (optional)

You can install the dependencies using pip:
- pip install torch tqdm

2. Run the training script

To train the AI, simply run the train.py script:
- python train.py

This will start the training process and print the cumulative reward every 100 epochs.

## Some concepts
1. Proximal Policy Optimization (PPO)

PPO is a reinforcement learning algorithm that improves the policy (the AI's strategy) by optimizing a clipped surrogate objective. It ensures stable updates by limiting the change in the policy at each step.

Key Components of PPO:

- clipped surrogate objective: encourages stable policy updates by clipping the probability ratio

- value function loss: ensures accurate estimation of the value of states

- entropy bonus: encourages exploration by rewarding higher entropy in the policy

2. Tic-Tac-Toe Environment

The Game class simulates a Tic-Tac-Toe game. The AI interacts with this environment to learn how to play the game effectively.

## Customization
1. Hyperparameters

You can adjust the following hyperparameters in train.py:

Learning Rates:

- actor_optimizer: learning rate for the Actor (default: 1e-4)

- critic_optimizer: learning rate for the Critic (default: 1e-3)

- clip parameter: controls the clipping range for the surrogate objective (default: 0.2)
  
- entropy bonus weight: controls the strength of the entropy bonus (default: 0.01)

2. Training Duration

You can adjust the number of training epochs by modifying the epochs parameter in the train function:

- def train(epochs=1000):
