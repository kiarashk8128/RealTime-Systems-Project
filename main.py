import pandas as pd
import numpy as np
from utils import helper_functions
from dqn import DQNAgent, JobshopEnvironment, train_dqn
import random
from pprint import pprint

# Number of machines and jobs
number_of_machines = 6
number_of_jobs = 15

# Parameters for deadline and energy consumption
alpha = 4  # Deadline multiplier
beta = 0.1  # Penalty multiplier for energy consumption
W1 = 1.2  # Weight for makespan in the reward function
W2 = 2.1  # Weight for energy consumption in the reward function


def main():
    tasks = helper_functions.generate_samlpe_tasks(number_of_jobs, number_of_machines)
    print(tasks)

    # Data structure to store results
    data = {
        "DQN": {
            "AvgWaitingTime1": [], "Waiting1": [], "AvgWaitingTime2": [], "Waiting2": [], "ResponseTime1": [],
            "ResponseTime2": [], "AvgResponseTime1": [], "AvgResponseTime2": [], 'Makespan1': [], "Makespan2": [],
            "MissedDeadlines1": [], "MissedDeadlines2": [], "TotalEnergy1": [], "TotalEnergy2": []
        },
    }

    # n constant, m variable
    list1 = []
    for m in range(2, number_of_machines + 1):
        sub_table = [row[:m + 1] for row in tasks]

        # DQN Algorithm
        env = JobshopEnvironment(sub_table, m, alpha, beta, W1, W2)
        agent = DQNAgent(state_size=env.state_size, action_size=env.action_size)
        train_dqn(agent, env, episodes=100)  # Adjust episodes as needed

        # Evaluate the trained DQN agent
        state = env.reset()
        total_reward = 0
        while True:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            state = next_state
            total_reward += reward
            if done:
                break

        # Assuming reward is negative of makespan
        dqn_makespan = -total_reward
        dqn_end_times = env.get_end_times()
        dqn_start_times = env.get_start_times()

        dqn_waiting_times = env.calculate_waiting_time(dqn_start_times)
        dqn_avg_waiting_time = np.mean(dqn_waiting_times)
        dqn_response_times = env.calculate_response_time(dqn_end_times)
        dqn_avg_response_time = np.mean(dqn_response_times)
        missed_deadlines, total_energy = env.get_metrics()

        data["DQN"]["AvgWaitingTime1"].append(dqn_avg_waiting_time)
        data["DQN"]["Waiting1"].append(dqn_waiting_times)
        data["DQN"]["ResponseTime1"].append(dqn_response_times)
        data["DQN"]["AvgResponseTime1"].append(dqn_avg_response_time)
        data["DQN"]["Makespan1"].append(dqn_makespan)
        data["DQN"]["MissedDeadlines1"].append(missed_deadlines)
        data["DQN"]["TotalEnergy1"].append(total_energy)

    pprint(data)

    # m constant, n variable
    list2 = []
    counter = 1
    for n in range(2, number_of_jobs + 1):
        counter += 1
        sub_table = tasks[:n]

        # DQN Algorithm
        env = JobshopEnvironment(sub_table, number_of_machines, alpha, beta, W1, W2)
        agent = DQNAgent(state_size=env.state_size, action_size=env.action_size)
        train_dqn(agent, env, episodes=100)  # Adjust episodes as needed

        # Evaluate the trained DQN agent
        state = env.reset()
        total_reward = 0
        while True:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            state = next_state
            total_reward += reward
            if done:
                break

        # Assuming reward is negative of makespan
        dqn_makespan = -total_reward
        dqn_end_times = env.get_end_times()
        dqn_start_times = env.get_start_times()

        dqn_waiting_times = env.calculate_waiting_time(dqn_start_times)
        dqn_avg_waiting_time = np.mean(dqn_waiting_times)
        dqn_response_times = env.calculate_response_time(dqn_end_times)
        dqn_avg_response_time = np.mean(dqn_response_times)
        missed_deadlines, total_energy = env.get_metrics()

        data["DQN"]["AvgWaitingTime2"].append(dqn_avg_waiting_time)
        data["DQN"]["Waiting2"].append(dqn_waiting_times)
        data["DQN"]["ResponseTime2"].append(dqn_response_times)
        data["DQN"]["AvgResponseTime2"].append(dqn_avg_response_time)
        data["DQN"]["Makespan2"].append(dqn_makespan)
        data["DQN"]["MissedDeadlines2"].append(missed_deadlines)
        data["DQN"]["TotalEnergy2"].append(total_energy)

    pprint(data)
    helper_functions.plot_results(data, number_of_machines, number_of_jobs)
    helper_functions.plot_resultts(data, number_of_machines, number_of_jobs)


if __name__ == "__main__":
    main()
