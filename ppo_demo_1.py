import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as distributions

import matplotlib.pyplot as plt
import numpy as np
import random

from drone_control_gym import * # Import your drone environment
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(log_dir="saves/ppo_demo1/")

# Set random seed for reproducibility
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Create train and test environments using your DroneControlGym
train_env = DroneControlGym()
test_env = DroneControlGym()

# MLP for Actor and Critic
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Dropout(dropout),
            nn.PReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout),
            nn.PReLU(),
            nn.Linear(hidden_dim, output_dim)  # Output is the number of actions (16)
        )

    def forward(self, x):
        return self.net(x)

# Actor-Critic Network (Actor for actions, Critic for value estimation)
class ActorCritic(nn.Module):
    def __init__(self, actor, critic):
        super().__init__()
        self.actor = actor
        self.critic = critic

    def forward(self, state):
        action_pred = self.actor(state)  # Predict action probabilities
        value_pred = self.critic(state)  # Predict state value
        return action_pred, value_pred

    def choose_action(self, state):
        # Convert state to tensor, ensuring it's in the correct format (e.g., NumPy array)
        if isinstance(state, np.ndarray):
            state_tensor = torch.FloatTensor(state).unsqueeze(0)  # Convert state to tensor
        elif isinstance(state, list):
            state_tensor = torch.FloatTensor(np.array(state)).unsqueeze(0)
        else:
            raise ValueError(f"Expected state to be a sequence, got {type(state)}")

        action_pred, value_pred = self(state_tensor)  # Get action prediction and value estimate

        # Apply softmax to get action probabilities
        action_prob = F.softmax(action_pred, dim=-1)
        dist = distributions.Categorical(action_prob)  # Create a categorical distribution

        # Sample an action
        action = dist.sample()
        log_prob_action = dist.log_prob(action)  # Get the log probability of the action

        return action, log_prob_action, value_pred

INPUT_DIM = 17  # = 17, as derived from get_all_state()

# Define the number of actions based on your drone's action list
OUTPUT_DIM = len(ACTIONS)  # There are 16 actions in the ACTIONS list

HIDDEN_DIM = 256  # You can adjust this based on your model's complexity

# Create the actor and critic networks
actor = MLP(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)
critic = MLP(INPUT_DIM, HIDDEN_DIM, 1)

# Combine them into the ActorCritic model
policy = ActorCritic(actor, critic)

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0)

policy.apply(init_weights)
LEARNING_RATE = 0.0005

optimizer = optim.Adam(policy.parameters(), lr = LEARNING_RATE)

def train(env, agent, optimizer, discount_factor, trace_decay, max_steps=500):
    policy.train()
    
    log_prob_actions = []
    values = []
    rewards = []
    dones = []
    episode_reward = 0
    step_count = 0
    done = False

    
    # Reset environment
    state = env.reset()

    while not done and step_count < max_steps:
        # Choose action using agent's method
        action, log_prob_action, value_pred = agent.choose_action(state)
        
        # Take action in the environment using predefined action list (ACTIONS)
        action_array = ACTIONS[action.item()]  # Map integer action to actual action array
        reward, done, state = env.step(action_array)  # Pass mapped action to the environment
        # Store log prob, value, reward, and done flags
        log_prob_actions.append(log_prob_action)
        values.append(value_pred)
        rewards.append(reward)
        dones.append(done)

        # Update episode reward and step count
        episode_reward += reward
        step_count += 1

    # Convert lists to tensors
    log_prob_actions = torch.cat(log_prob_actions)
    values = torch.cat(values).squeeze(-1)
    
    # Calculate returns and advantages
    returns = calculate_returns(rewards, discount_factor)
    advantages = calculate_advantages(rewards, values, discount_factor, trace_decay)
    
    # Update policy and value networks
    policy_loss, value_loss = update_policy(advantages, log_prob_actions, returns, values, optimizer)
    
    # Log policy and value loss to TensorBoard
    writer.add_scalar("Loss/Policy Loss", policy_loss, step_count)  # Log policy loss
    writer.add_scalar("Loss/Value Loss", value_loss, step_count)    # Log value loss


    return policy_loss, value_loss, episode_reward, step_count

def calculate_returns(rewards, discount_factor, normalize=True):
    returns = []
    R = 0
    for r in reversed(rewards):
        R = r + R * discount_factor
        returns.insert(0, R)
    returns = torch.tensor(returns)
    if normalize and len(returns) > 1:
        # Add epsilon to avoid division by zero
        returns = (returns - returns.mean()) / (returns.std() + 1e-5)
    return returns

def calculate_advantages(rewards, values, discount_factor, trace_decay, normalize=True):
    advantages = []
    advantage = 0
    next_value = 0
    for r, v in zip(reversed(rewards), reversed(values)):
        td_error = r + next_value * discount_factor - v
        advantage = td_error + advantage * discount_factor * trace_decay
        next_value = v
        advantages.insert(0, advantage)
    advantages = torch.tensor(advantages)
    if normalize and len(advantages) > 1:
        # Add epsilon to avoid division by zero
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)
    return advantages

def update_policy(advantages, log_prob_actions, returns, values, optimizer):
    advantages = advantages.detach()
    returns = returns.detach()

    # Calculate the policy loss
    policy_loss = -(advantages * log_prob_actions).sum()

    # Calculate the value loss using Huber loss (smooth L1 loss)
    value_loss = F.smooth_l1_loss(returns, values).sum()

    # Backpropagate and optimize
    optimizer.zero_grad()
    policy_loss.backward()
    value_loss.backward()
    optimizer.step()

    return policy_loss.item(), value_loss.item()

def evaluate(env, policy, max_steps=500):
    policy.eval()
    
    episode_reward = 0
    done = False
    step_count = 0

    state = env.reset()

    while not done and step_count < max_steps:
        state = torch.FloatTensor(state).unsqueeze(0)

        with torch.no_grad():
            action_pred, _ = policy(state)
            action_prob = F.softmax(action_pred, dim=-1)

        # Ensure numerical stability by checking if action_prob contains NaNs or invalid values
        if torch.isnan(action_prob).any():
            print("Invalid action probabilities detected, skipping step.")
            break

        action = torch.argmax(action_prob, dim=-1)

        # Map action index to actual action array using the ACTIONS mapping
        action_array = ACTIONS[action.item()]  # Get the corresponding action array
        
        # Pass the mapped action array to the environment's step function
        reward, done, state = env.step(action_array)  # Use the correct action array format
        
        episode_reward += reward
        step_count += 1

    return episode_reward, step_count

def save_model(policy, optimizer, filename="saves/ppo_demo1_drone_model.pth"):
    checkpoint = {
        'policy_state_dict': policy.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    torch.save(checkpoint, filename)
    print(f"Model and optimizer saved to {filename}")

def load_model(policy, optimizer, filename="saves/ppo_demo1_drone_model.pth"):
    checkpoint = torch.load(filename)
    policy.load_state_dict(checkpoint['policy_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f"Model and optimizer loaded from {filename}")


LOAD_MODEL = False
MAX_EPISODES = 5000
DISCOUNT_FACTOR = 0.99
TRACE_DECAY = 0.99
N_TRIALS = 100
MAX_STEPS = 500  # Add max steps limit to prevent the agent from running indefinitely

train_rewards = []
test_rewards = []

start_step = 1
if LOAD_MODEL:
    load_model()
    start_step = int(input("Enter the starting step: "))
    print("Starting from step", start_step)

for episode in range(start_step, MAX_EPISODES + 1):
    
    print(f'Episode {episode}')
    # Train the policy using the drone environment
    policy_loss, value_loss, train_reward, train_step_count = train(train_env, policy, optimizer, DISCOUNT_FACTOR, TRACE_DECAY)
    print(f'Train Reward: {train_reward:.2f} | Training Steps: {train_step_count}')
    # Evaluate the policy in test mode (drone environment)
    test_reward, test_step_count = evaluate(test_env, policy, max_steps=MAX_STEPS)  # Adding max_steps for evaluation
    print(f'Test Reward: {test_reward:.2f} | Testing Steps: {test_step_count}')
    
    train_rewards.append(train_reward)
    test_rewards.append(test_reward)
    
    mean_train_rewards = np.mean(train_rewards[-N_TRIALS:])
    mean_test_rewards = np.mean(test_rewards[-N_TRIALS:])
    
    # Log train and test rewards to TensorBoard
    writer.add_scalar('Train/Reward', mean_train_rewards, episode)
    writer.add_scalar('Test/Reward', mean_test_rewards, episode)
    writer.add_scalar('Loss/Policy Loss', policy_loss, episode)
    writer.add_scalar('Loss/Value Loss', value_loss, episode)
    
    print(f'Mean Train Reward: {mean_train_rewards:.2f} | Mean Test Reward: {mean_test_rewards:.2f} ')
    print("\n")
    
    if episode % 10 == 0:
        save_model(policy, optimizer, filename=f"saves/ppo_demo1_drone_model.pth")

plt.figure(figsize=(12,8))
plt.plot(test_rewards, label='Test Reward')
plt.plot(train_rewards, label='Train Reward')
plt.xlabel('Episode', fontsize=20)
plt.ylabel('Reward', fontsize=20)
plt.legend(loc='lower right')
plt.grid()
plt.show()
writer.close()
