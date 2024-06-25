import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque


class FlowshopEnvironment:
    def __init__(self, tasks, num_machines):
        self.tasks = tasks
        self.num_machines = num_machines
        self.state_size = len(tasks) * num_machines
        self.action_size = len(tasks)
        self.reset()

    def reset(self):
        self.schedule = []
        self.time_elapsed = 0
        self.current_task = 0
        self.state = np.zeros(self.state_size)
        self.machines_free_time = [0] * self.num_machines
        self.task_start_times = np.zeros((len(self.tasks), self.num_machines))
        self.task_end_times = np.zeros((len(self.tasks), self.num_machines))
        return self.state

    def step(self, action):
        task = self.tasks[action]
        self.schedule.append(task)
        self.current_task += 1

        start_time = max(self.time_elapsed, self.machines_free_time[0])
        self.task_start_times[action][0] = start_time
        self.machines_free_time[0] = start_time + task[1]

        for machine in range(1, self.num_machines):
            start_time = max(self.machines_free_time[machine - 1], self.machines_free_time[machine])
            self.task_start_times[action][machine] = start_time
            self.machines_free_time[machine] = start_time + task[machine + 1]
            self.time_elapsed = max(self.time_elapsed, self.machines_free_time[machine])

        self.task_end_times[action] = self.machines_free_time

        self.state = self.get_state()
        reward = -self.time_elapsed
        done = self.current_task == len(self.tasks)
        if done:
            reward += 1000

        return self.state, reward, done, {}

    def get_state(self):
        state = np.zeros(self.state_size)
        for i, task in enumerate(self.schedule):
            for j in range(self.num_machines):
                state[i * self.num_machines + j] = task[j + 1]
        return state

    def get_end_times(self):
        return self.task_end_times

    def get_start_times(self):
        return self.task_start_times

    def calculate_waiting_time(self, start_times):
        num_jobs = start_times.shape[0]
        waiting_times = [start_times[i][0] for i in range(num_jobs)]
        return waiting_times

    def calculate_response_time(self, end_times):
        num_jobs = end_times.shape[0]
        response_times = [end_times[i][-1] for i in range(num_jobs)]
        return response_times


# Define the Q-Network
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# Define the agent
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.learning_rate = 0.001
        self.model = QNetwork(state_size, action_size)
        self.target_model = QNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.update_target_model()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0)
        act_values = self.model(state)
        return torch.argmax(act_values, dim=1).item()

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = torch.FloatTensor(state).unsqueeze(0)
            next_state = torch.FloatTensor(next_state).unsqueeze(0)
            target = reward
            if not done:
                target += self.gamma * torch.max(self.target_model(next_state)).item()
            target_f = self.model(state)
            target_f[0][action] = target
            self.optimizer.zero_grad()
            loss = nn.MSELoss()(self.model(state), target_f)
            loss.backward()
            self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


# Training the DQN Agent
def train_dqn(agent, env, episodes=1000, batch_size=32):
    for e in range(episodes):
        state = env.reset()
        for time in range(500):
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                agent.update_target_model()
                print(f"Episode {e + 1}/{episodes}, Score: {time}, Epsilon: {agent.epsilon:.2}")
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
