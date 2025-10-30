
import torch
import gymnasium as gym
import torch.nn as nn
import torch.nn.functional as F

from utils import compute_advantage, train_on_policy_agent, set_seed

class PolicyNet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        hidden = F.relu(self.fc1(state))
        action = F.softmax(self.fc2(hidden), dim=-1)
        return action   
    
class ValueNet(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        hidden = F.relu(self.fc1(state))
        value = self.fc2(hidden)
        return value

class PPO:

    def __init__(self, state_dim, hidden_dim, action_dim,
                 actor_lr, critic_lr, lmbda, epochs,
                 eps, gamma, device):
        # NOTE: 
        # 1. actor takes in state, output action;
        # 2. critic takes in state, output value of that state
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.device = device
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs
        self.eps = eps

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float32).to(self.device)
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample().item()
        return action
    
    def update(self, transition_dict):
        """
        After a episode, agent will call `update` interface:
        """
        states = torch.tensor(transition_dict['states'],
                                dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(
                                self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                                dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],
                                dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                                dtype=torch.float).view(-1, 1).to(self.device)

        # NOTE:
        # 1. compute advantage through GAE
        # 2. a trajectory contains many (state, action, next_state, reward) pairs for training
        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        td_error = td_target - self.critic(states)
        # NOTE: In compute_advantage function, td_error is detached
        advantage = compute_advantage(self.gamma, self.lmbda, td_error.cpu()).to(self.device)

        behaviro_policy_prob = torch.log(self.actor(states).gather(1, actions)).detach()

        # Use the same training data (sampled data) to train our networks for many times
        for step_id in range(self.epochs):
            target_policy_prob = torch.log(self.actor(states).gather(1, actions))
            ration = torch.exp(target_policy_prob - behaviro_policy_prob)
            surr1 = ration * advantage
            surr2 = torch.clamp(ration, 1 - self.eps, 1 + self.eps) * advantage

            # Update actor (Policy network)
            actor_loss = -torch.min(surr1, surr2).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update critic (Value network)
            critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()


def train_car_pole():
    set_seed(seed=42)

    # learning rate
    actor_lr = 1e-3
    critic_lr = 1e-2

    num_episodes = 500      # sampled trajectory number
    gamma = 0.98            # discount factor
    lmbda = 0.95            # ajusts the bias-variance tradeoff, larger lmbda, larger variance, smaller bias
    epochs = 10             # each data sample is used for epochs time
    eps = 0.2
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
        "cpu")

    # Environment setup
    render = True
    env = gym.make('CartPole-v1', render_mode='human' if render else None)

    # Network params
    state_dim = env.observation_space.shape[0]
    hidden_dim = 128
    action_dim = env.action_space.n
    agent = PPO(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda,
                epochs, eps, gamma, device)
    return_list = train_on_policy_agent(env, agent, num_episodes)

if __name__ == "__main__":
    train_car_pole()