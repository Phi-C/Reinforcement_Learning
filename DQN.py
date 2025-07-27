import random
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import collections
import wandb
from tqdm import tqdm

def setup_simulation(name, render=True, seed=0):
    """
    Sets up the simulation environment for the DQN algorithm.
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    env = gym.make('CartPole-v1', render_mode='human' if render else None)
    # env.seed(seed)
    return env


class ReplayBuffer(object):
    def __init__(self, capacity):
        super(ReplayBuffer, self).__init__()
        self.capacity = capacity
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        """添加一条经验([S_t, A_t, R_{t+1}, S_{t+1}])到缓冲区"""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        samples = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*samples)
        return np.array(states), actions, rewards, np.array(next_states), dones

    @property
    def size(self):
        return len(self.buffer)

class QNet(nn.Module):
    """
        QNetwork for Deep Q-Learning.
        This network takes the state as input and outputs Q-values for each action. Q(a)
    """

    def __init__(self, state_dim, hidden_dim, action_dim):
        super(QNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        """
            state: 输入状态, shape为(batch_size, state_dim)
            返回: 输出每个动作的Q值, shape为(batch_size, action_dim)
        """
        x = F.relu(self.fc1(state))
        return self.fc2(x)


class DQN(object):

    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate,
                gamma, epsilon, target_update, device):
        super(DQN, self).__init__()

        # Q-Network: 输入状态, 输出每个动作的Q值
        self.q_net = QNet(state_dim, hidden_dim, action_dim).to(device)

        # target_q_net隔若干步才会和q_net同步一次参数, 这样可以解决训练稳定性问题
        self.target_q_net = QNet(state_dim, hidden_dim, action_dim).to(device)

        self.action_dim = action_dim

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate)
        self.gamma = gamma                 # discounted reward factor
        self.epsilon = epsilon              # ε-greedy策略的ε值
        self.target_update = target_update  # 更新目标网络的频率
        self.count = 0
        self.device = device

    def select_action(self, state):
        """
        根据当前状态选择动作, 使用ε-greedy策略
        """
        if random.random() < self.epsilon:
            action = random.randint(0, self.action_dim - 1)
        else:
            # TODO: check
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_net(state)
            action = q_values.argmax().item()
        
        return action

    def update(self, transition_dic):
        states = torch.tensor(transition_dic['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dic['actions'], dtype=torch.long).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dic['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dic['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dic['dones'], dtype=torch.float).view(-1, 1).to(self.device)

        q_values = self.q_net(states).gather(1, actions)

        max_next_q_values = self.target_q_net(next_states).max(1)[0].view(-1, 1)

        # TD-error
        q_trgets = rewards + self.gamma * max_next_q_values * (1 - dones)

        dqn_loss = torch.mean(F.mse_loss(q_values, q_trgets))
        self.optimizer.zero_grad()
        dqn_loss.backward()
        self.optimizer.step()

        # 更新目标网络
        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())

        self.count += 1
        

class Trainer(object):
    def __init__(self,
                env,
                dqn,
                buffer_size=10000,
                minimal_size=500,
                batch_size=64,
                max_episodes=500,
                max_steps=10):
        self.env = env
        self.dqn = dqn

        self.buffer = ReplayBuffer(buffer_size)
        self.minimal_size = minimal_size
        self.batch_size = batch_size

        self.max_episodes = max_episodes
        self.max_steps = max_steps
        wandb.init(project="DQN-CartPole", name="RL_exp", settings=wandb.Settings(init_timeout=120))  # Replace with your WandB entity name

    def train(self):
        return_list = []
        for i in range(self.max_steps):
            with tqdm(total=int(self.max_episodes / self.max_steps), desc="Iteration % i") as pbar:
                for episode_idx in range(int(self.max_episodes / self.max_steps)):
                    episode_return = 0
                    # 重置环境
                    state = self.env.reset()[0]
                    done = False
                    while not done:
                        # 选择动作
                        action = self.dqn.select_action(state)
                        # 执行动作
                        # import pdb; pdb.set_trace()
                        next_state, reward, done, _, _ = self.env.step(action)
                        # 存储经验到缓冲区
                        self.buffer.add(state, action, reward, next_state, done)
                        state = next_state
                        episode_return += reward

                        # 当buffer中的数据量超过一定值后, 才进行QNet的训练
                        if self.buffer.size >= self.minimal_size:
                            b_state, b_action, b_reward, b_next_state, b_done = self.buffer.sample(self.batch_size)
                            transition_dic = {
                                'states': b_state,
                                'actions': b_action,
                                'rewards': b_reward,
                                'next_states': b_next_state,
                                'dones': b_done
                            }
                            self.dqn.update(transition_dic)
                    
                    return_list.append(episode_return)

                    wandb.log({
                        'episode': self.max_episodes / self.max_steps * i + episode_idx + 1,
                        'return': episode_return
                    })
                    if (episode_idx + 1) % 10 == 0:
                        pbar.set_postfix({
                            'episode':
                            '%d' % (self.max_episodes / self.max_steps * i + episode_idx + 1),
                            'return':
                            '%.3f' % np.mean(return_list[-10:])
                        })
                    pbar.update(1)

        self.env.close()

                

if __name__ == "__main__":
    # 设置参数
    name = 'CartPole-v1'
    hidden_dim = 128
    learning_rate = 0.002
    gamma = 0.98
    epsilon = 0.01
    target_update = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = setup_simulation(name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    dqn = DQN(state_dim, hidden_dim, action_dim, learning_rate, gamma, epsilon, target_update, device)
    
    trainer = Trainer(env, dqn)
    trainer.train()