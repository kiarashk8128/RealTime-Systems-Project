from algorithms import genetic
from algorithms import johnson
import pandas as pd
import numpy as np
from utils import helper_functions
from dqn import DQNAgent, FlowshopEnvironment, train_dqn
import random
from pprint import pprint

number_of_machines = 4
number_of_jobs = 20


def main():
    tasks = helper_functions.generate_samlpe_tasks(number_of_jobs, number_of_machines)
    print(tasks)

    # delay1 & waiting1 --> refers to n constant and m variable
    # Delay2 & Waiting2 --> refers to m constant and n variable
    data = {
        "Johnson": {"AvgWaitingTime1": [], "Waiting1": [], "AvgWaitingTime2": [], "Waiting2": [], "ResponseTime1": [],
                    "ResponseTime2": [], "AvgResponseTime1": [], "AvgResponseTime2": [], 'Makespan1': [],
                    "Makespan2": []},
        "DQN": {"AvgWaitingTime1": [], "Waiting1": [], "AvgWaitingTime2": [], "Waiting2": [], "ResponseTime1": [],
                "ResponseTime2": [], "AvgResponseTime1": [], "AvgResponseTime2": [], 'Makespan1': [], "Makespan2": []}
        "Genetic": {"AvgWaitingTime1": [], "Waiting1": [], "AvgWaitingTime2": [], "Waiting2": [], "ResponseTime1": [],
                    "ResponseTime2": [], "AvgResponseTime1": [], "AvgResponseTime2": [], 'Makespan1': [],
                    "Makespan2": []}
    }

    # n constant , m variable
    list1 = []
    for m in range(2, number_of_machines + 1):
        # Johnson
        sub_table = [row[:m + 1] for row in tasks]
        j = johnson.JohnsonAlgorithm(sub_table)
        ordered_jobs = j.johnsons_algorithm()
        print(ordered_jobs)
        makespan, end_times, start_times = j.calculate_makespan(ordered_jobs)
        waiting_times = j.calculate_waiting_time(ordered_jobs, start_times)
        avg_waiting_time = j.calculate_average_waiting_time(waiting_times)
        response_times = j.calculate_response_time(ordered_jobs, end_times)
        avg_response_time = j.calculate_average_response_time(response_times)

        data["Johnson"]["AvgWaitingTime1"].append(avg_waiting_time)
        data["Johnson"]["Waiting1"].append(waiting_times)
        data["Johnson"]["ResponseTime1"].append(response_times)
        data["Johnson"]["AvgResponseTime1"].append(avg_response_time)
        data["Johnson"]["Makespan1"].append(makespan)

        #Genetic
        g = genetic.GeneticAlgorithm(sub_table)
        best_sequence = g.run()
        print("#######################################")
        print(best_sequence)
        makespan, end_times, start_times = g.calculate_makespan(best_sequence)
        waiting_times = g.calculate_waiting_time(best_sequence, start_times)
        avg_waiting_time = g.calculate_average_waiting_time(waiting_times)
        response_times = g.calculate_response_time(best_sequence, end_times)
        avg_response_time = g.calculate_average_response_time(response_times)

        data["Genetic"]["AvgWaitingTime1"].append(avg_waiting_time)
        data["Genetic"]["Waiting1"].append(waiting_times)
        data["Genetic"]["ResponseTime1"].append(response_times)
        data["Genetic"]["AvgResponseTime1"].append(avg_response_time)
        data["Genetic"]["Makespan1"].append(makespan)

        # DQN Algorithm
        env = FlowshopEnvironment(sub_table, m)
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

        data["DQN"]["AvgWaitingTime1"].append(dqn_avg_waiting_time)
        data["DQN"]["Waiting1"].append(dqn_waiting_times)
        data["DQN"]["ResponseTime1"].append(dqn_response_times)
        data["DQN"]["AvgResponseTime1"].append(dqn_avg_response_time)
        data["DQN"]["Makespan1"].append(dqn_makespan)

    pprint(data)
    # m constant , n variable
    list2 = []
    counter = 1
    for n in range(2, number_of_jobs + 1):
        counter += 1
        sub_table = tasks[:n]
        # Johnson Algorithm
        j = johnson.JohnsonAlgorithm(sub_table)
        ordered_jobs = j.johnsons_algorithm()
        makespan, end_times, start_times = j.calculate_makespan(ordered_jobs)

        waiting_times = j.calculate_waiting_time(ordered_jobs, start_times)
        avg_waiting_time = j.calculate_average_waiting_time(waiting_times)
        response_times = j.calculate_response_time(ordered_jobs, end_times)
        avg_response_time = j.calculate_average_response_time(response_times)

        data["Johnson"]["AvgWaitingTime2"].append(avg_waiting_time)
        data["Johnson"]["Waiting2"].append(waiting_times)
        data["Johnson"]["ResponseTime2"].append(response_times)
        data["Johnson"]["AvgResponseTime2"].append(avg_response_time)
        data["Johnson"]["Makespan2"].append(makespan)

        # Genetic
        g = genetic.GeneticAlgorithm(sub_table)
        ordered_jobs = g.run()
        makespan, end_times, start_times = g.calculate_makespan(ordered_jobs)

        waiting_times = g.calculate_waiting_time(ordered_jobs, start_times)
        avg_waiting_time = g.calculate_average_waiting_time(waiting_times)
        response_times = g.calculate_response_time(ordered_jobs, end_times)
        avg_response_time = g.calculate_average_response_time(response_times)

        data["Genetic"]["AvgWaitingTime2"].append(avg_waiting_time)
        data["Genetic"]["Waiting2"].append(waiting_times)
        data["Genetic"]["ResponseTime2"].append(response_times)
        data["Genetic"]["AvgResponseTime2"].append(avg_response_time)
        data["Genetic"]["Makespan2"].append(makespan)
        # DQN Algorithm
        env = FlowshopEnvironment(sub_table, number_of_machines)
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

        data["DQN"]["AvgWaitingTime2"].append(dqn_avg_waiting_time)
        data["DQN"]["Waiting2"].append(dqn_waiting_times)
        data["DQN"]["ResponseTime2"].append(dqn_response_times)
        data["DQN"]["AvgResponseTime2"].append(dqn_avg_response_time)
        data["DQN"]["Makespan2"].append(dqn_makespan)

    pprint(data)
    helper_functions.plot_results(data, number_of_machines, number_of_jobs)

if __name__ == "__main__":
    main()
