# %%
import random#随机数
import numpy as np
import torch#深度学习框架
import torch.nn.functional as F
import matplotlib.pyplot as plt
import rl_utils
import collections

class Qnet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Qnet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class DQN:
    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate, gamma,
                 epsilon, target_update, device):
        self.action_dim = action_dim
        self.q_net = Qnet(state_dim, hidden_dim,self.action_dim).to(device)  # Q网络
        self.target_q_net = Qnet(state_dim, hidden_dim,self.action_dim).to(device)  # 目标网络
        self.optimizer = torch.optim.Adam(self.q_net.parameters(),lr=learning_rate)  # 使用Adam优化器
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # epsilon-贪婪策略
        self.target_update = target_update  # 目标网络更新频率
        self.count = 0  # 计数器,记录更新次数
        self.device = device

    def take_action(self, state):#选择动作，epsilon-greedy策略
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            state = torch.tensor([state], dtype=torch.float).to(self.device)
            action = self.q_net(state).argmax().item()#argmax()返回最大值的索引
        return action

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'],dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions'],dtype=torch.float).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'],dtype=torch.float).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],dtype=torch.float).to(self.device)

        q_values = max(self.q_net(states))#当前状态的Q值
        # 下个状态的最大Q值
        max_next_q_values = max(self.target_q_net(next_states))
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)  # TD误差目标
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))  # 均方误差损失函数
        self.optimizer.zero_grad()  # PyTorch中默认梯度会累积,这里需要显式将梯度置为0
        dqn_loss.backward()
        self.optimizer.step()

        if self.count % self.target_update == 0:  # 这是判断是否更新目标网络的条件：每隔target_update次更新一次
            self.target_q_net.load_state_dict(self.q_net.state_dict())  # 更新目标网络
        self.count += 1
