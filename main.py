from algorithms import genetic
from algorithms import johnson
import pandas as pd
import numpy as np
from utils import helper_functions
from utils.helper_functions import calculate_missed_deadlines_and_energy
from dqn import DQNAgent, JobshopEnvironment, train_dqn
import random
from pprint import pprint


number_of_machines = 5
number_of_jobs = 12



alpha = 1.5  # Deadline multiplier
beta = 0.1  # Penalty multiplier for energy consumption
W1 = 4.9
W2 = 8.2


def main():
    # number_of_jobs and number_of_machines would be defined earlier in your script
    # at below, you can see the complete random code for task generating scenario, if you need it, uncomment it.
    # tasks = helper_functions.generate_samlpe_tasks(number_of_jobs, number_of_machines)
    # deadlines = [alpha * sum(task[1:]) for task in tasks]
    # base_energy = [np.random.uniform(10, 20) for _ in range(number_of_jobs)]
    tasks, base_energy, selected_operations = helper_functions.generate_real_data_sample_tasks(number_of_jobs,
                                                                                               number_of_machines)

    # Extract task times for deadlines
    deadlines = [alpha * sum(task[1:]) for task in tasks]  # No change needed here

    print("Selected operations distribution:")
    pprint(selected_operations)
    data = {
            "Johnson": {"AvgWaitingTime1": [], "Waiting1": [], "AvgWaitingTime2": [], "Waiting2": [], "ResponseTime1": [],
                        "ResponseTime2": [], "AvgResponseTime1": [], "AvgResponseTime2": [], 'Makespan1': [],
                        "Makespan2": [], "MissedDeadlines1": [], "MissedDeadlines2": [], "TotalEnergy1": [],
                        "TotalEnergy2": []},
            "DQN": {"AvgWaitingTime1": [], "Waiting1": [], "AvgWaitingTime2": [], "Waiting2": [], "ResponseTime1": [],
                    "ResponseTime2": [], "AvgResponseTime1": [], "AvgResponseTime2": [], 'Makespan1': [], "Makespan2": [],
                    "MissedDeadlines1": [], "MissedDeadlines2": [], "TotalEnergy1": [], "TotalEnergy2": []},
            "Genetic": {"AvgWaitingTime1": [], "Waiting1": [], "AvgWaitingTime2": [], "Waiting2": [], "ResponseTime1": [],
                        "ResponseTime2": [], "AvgResponseTime1": [], "AvgResponseTime2": [], 'Makespan1': [],
                        "Makespan2": [], "MissedDeadlines1": [], "MissedDeadlines2": [], "TotalEnergy1": [],
                        "TotalEnergy2": []}
        }

    # n constant , m variable
    for m in range(2, number_of_machines + 1):
        sub_table = [row[:m + 1] for row in tasks]

        # Johnson Algorithm
        j = johnson.JohnsonAlgorithm(sub_table)
        ordered_jobs = j.johnsons_algorithm()
        makespan, end_times, start_times = j.calculate_makespan(ordered_jobs)
        waiting_times = j.calculate_waiting_time(ordered_jobs, start_times)
        avg_waiting_time = j.calculate_average_waiting_time(waiting_times)
        response_times = j.calculate_response_time(ordered_jobs, end_times)
        avg_response_time = j.calculate_average_response_time(response_times)

        missed_deadlines, total_energy = calculate_missed_deadlines_and_energy(
            ordered_jobs, end_times, deadlines, base_energy, beta
        )

        data["Johnson"]["AvgWaitingTime1"].append(avg_waiting_time)
        data["Johnson"]["Waiting1"].append(waiting_times)
        data["Johnson"]["ResponseTime1"].append(response_times)
        data["Johnson"]["AvgResponseTime1"].append(avg_response_time)
        data["Johnson"]["Makespan1"].append(makespan)
        data["Johnson"]["MissedDeadlines1"].append(missed_deadlines)
        data["Johnson"]["TotalEnergy1"].append(total_energy)

        # Genetic Algorithm
        g = genetic.GeneticAlgorithm(sub_table)
        best_sequence = g.run()
        makespan, end_times, start_times = g.calculate_makespan(best_sequence)
        waiting_times = g.calculate_waiting_time(best_sequence, start_times)
        avg_waiting_time = g.calculate_average_waiting_time(waiting_times)
        response_times = g.calculate_response_time(best_sequence, end_times)
        avg_response_time = g.calculate_average_response_time(response_times)

        missed_deadlines, total_energy = calculate_missed_deadlines_and_energy(
            best_sequence, end_times, deadlines, base_energy, beta
        )

        data["Genetic"]["AvgWaitingTime1"].append(avg_waiting_time)
        data["Genetic"]["Waiting1"].append(waiting_times)
        data["Genetic"]["ResponseTime1"].append(response_times)
        data["Genetic"]["AvgResponseTime1"].append(avg_response_time)
        data["Genetic"]["Makespan1"].append(makespan)
        data["Genetic"]["MissedDeadlines1"].append(missed_deadlines)
        data["Genetic"]["TotalEnergy1"].append(total_energy)

        # DQN Algorithm
        env = JobshopEnvironment(sub_table, m, alpha, beta, W1, W2)
        agent = DQNAgent(state_size=env.state_size, action_size=env.action_size)

        # Pre-training using Johnson's algorithm
        train_dqn(agent, env, ordered_jobs, episodes=100, batch_size=32,
                  pretrain_steps=50)  # Include pretrain_steps

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

        dqn_makespan, dqn_end_times, dqn_start_times = j.calculate_makespan(env.schedule)

        dqn_waiting_times = env.calculate_waiting_time(dqn_start_times)
        dqn_avg_waiting_time = np.mean(dqn_waiting_times)
        dqn_response_times = env.calculate_response_time(dqn_end_times)
        dqn_avg_response_time = np.mean(dqn_response_times)

        missed_deadlines, total_energy = calculate_missed_deadlines_and_energy(
            sub_table, dqn_end_times, deadlines, base_energy, beta
        )

        data["DQN"]["AvgWaitingTime1"].append(dqn_avg_waiting_time)
        data["DQN"]["Waiting1"].append(dqn_waiting_times)
        data["DQN"]["ResponseTime1"].append(dqn_response_times)
        data["DQN"]["AvgResponseTime1"].append(dqn_avg_response_time)
        data["DQN"]["Makespan1"].append(dqn_makespan)
        data["DQN"]["MissedDeadlines1"].append(missed_deadlines)
        data["DQN"]["TotalEnergy1"].append(total_energy)

    # m constant , n variable
    for n in range(2, number_of_jobs + 1):
        sub_table = tasks[:n]
        sub_deadlines = deadlines[:n]
        sub_base_energy = base_energy[:n]

        # Johnson Algorithm
        j = johnson.JohnsonAlgorithm(sub_table)
        ordered_jobs = j.johnsons_algorithm()
        makespan, end_times, start_times = j.calculate_makespan(ordered_jobs)
        waiting_times = j.calculate_waiting_time(ordered_jobs, start_times)
        avg_waiting_time = j.calculate_average_waiting_time(waiting_times)
        response_times = j.calculate_response_time(ordered_jobs, end_times)
        avg_response_time = j.calculate_average_response_time(response_times)

        missed_deadlines, total_energy = calculate_missed_deadlines_and_energy(
            ordered_jobs, end_times, sub_deadlines, sub_base_energy, beta
        )

        data["Johnson"]["AvgWaitingTime2"].append(avg_waiting_time)
        data["Johnson"]["Waiting2"].append(waiting_times)
        data["Johnson"]["ResponseTime2"].append(response_times)
        data["Johnson"]["AvgResponseTime2"].append(avg_response_time)
        data["Johnson"]["Makespan2"].append(makespan)
        data["Johnson"]["MissedDeadlines2"].append(missed_deadlines)
        data["Johnson"]["TotalEnergy2"].append(total_energy)

        # Genetic Algorithm
        g = genetic.GeneticAlgorithm(sub_table)
        best_sequence = g.run()
        makespan, end_times, start_times = g.calculate_makespan(best_sequence)
        waiting_times = g.calculate_waiting_time(best_sequence, start_times)
        avg_waiting_time = g.calculate_average_waiting_time(waiting_times)
        response_times = g.calculate_response_time(best_sequence, end_times)
        avg_response_time = g.calculate_average_response_time(response_times)

        missed_deadlines, total_energy = calculate_missed_deadlines_and_energy(
            best_sequence, end_times, sub_deadlines, sub_base_energy, beta
        )

        data["Genetic"]["AvgWaitingTime2"].append(avg_waiting_time)
        data["Genetic"]["Waiting2"].append(waiting_times)
        data["Genetic"]["ResponseTime2"].append(response_times)
        data["Genetic"]["AvgResponseTime2"].append(avg_response_time)
        data["Genetic"]["Makespan2"].append(makespan)
        data["Genetic"]["MissedDeadlines2"].append(missed_deadlines)
        data["Genetic"]["TotalEnergy2"].append(total_energy)

        env = JobshopEnvironment(sub_table, number_of_machines, alpha, beta, W1, W2)
        agent = DQNAgent(state_size=env.state_size, action_size=env.action_size)

        # Pre-training using Johnson's algorithm
        train_dqn(agent, env, ordered_jobs, episodes=100, batch_size=32,
                  pretrain_steps=50)  # Include pretrain_steps

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

        dqn_makespan, dqn_end_times, dqn_start_times = j.calculate_makespan(env.schedule)


        dqn_waiting_times = env.calculate_waiting_time(dqn_start_times)
        dqn_avg_waiting_time = np.mean(dqn_waiting_times)
        dqn_response_times = env.calculate_response_time(dqn_end_times)
        dqn_avg_response_time = np.mean(dqn_response_times)

        missed_deadlines, total_energy = calculate_missed_deadlines_and_energy(
            sub_table, dqn_end_times, sub_deadlines, sub_base_energy, beta
        )

        data["DQN"]["AvgWaitingTime2"].append(dqn_avg_waiting_time)
        data["DQN"]["Waiting2"].append(dqn_waiting_times)
        data["DQN"]["ResponseTime2"].append(dqn_response_times)
        data["DQN"]["AvgResponseTime2"].append(dqn_avg_response_time)
        data["DQN"]["Makespan2"].append(dqn_makespan)
        data["DQN"]["MissedDeadlines2"].append(missed_deadlines)
        data["DQN"]["TotalEnergy2"].append(total_energy)

    pprint(data)
    helper_functions.plot_results(data, number_of_machines, number_of_jobs)
    #
    # print('###################\n\n\n')
    # # at below, you can see the complete random code for task generating scenario, if you need it, uncomment it.
    tasks = helper_functions.generate_samlpe_tasks(number_of_jobs, number_of_machines)
    deadlines = [alpha * sum(task[1:]) for task in tasks]
    base_energy = [np.random.uniform(10, 20) for _ in range(number_of_jobs)]
    # tasks, base_energy, selected_operations = helper_functions.generate_real_data_sample_tasks(number_of_jobs,
    #                                                                                            number_of_machines)
    #
    # # Extract task times for deadlines
    # deadlines = [alpha * sum(task[1:]) for task in tasks]  # No change needed here
    #
    # print("Selected operations distribution:")
    # pprint(selected_operations)
    data = {
        "Johnson": {"AvgWaitingTime1": [], "Waiting1": [], "AvgWaitingTime2": [], "Waiting2": [], "ResponseTime1": [],
                    "ResponseTime2": [], "AvgResponseTime1": [], "AvgResponseTime2": [], 'Makespan1': [],
                    "Makespan2": [], "MissedDeadlines1": [], "MissedDeadlines2": [], "TotalEnergy1": [],
                    "TotalEnergy2": []},
        "DQN": {"AvgWaitingTime1": [], "Waiting1": [], "AvgWaitingTime2": [], "Waiting2": [], "ResponseTime1": [],
                "ResponseTime2": [], "AvgResponseTime1": [], "AvgResponseTime2": [], 'Makespan1': [], "Makespan2": [],
                "MissedDeadlines1": [], "MissedDeadlines2": [], "TotalEnergy1": [], "TotalEnergy2": []},
        "Genetic": {"AvgWaitingTime1": [], "Waiting1": [], "AvgWaitingTime2": [], "Waiting2": [], "ResponseTime1": [],
                    "ResponseTime2": [], "AvgResponseTime1": [], "AvgResponseTime2": [], 'Makespan1': [],
                    "Makespan2": [], "MissedDeadlines1": [], "MissedDeadlines2": [], "TotalEnergy1": [],
                    "TotalEnergy2": []}
    }

    # n constant , m variable
    for m in range(2, number_of_machines + 1):
        sub_table = [row[:m + 1] for row in tasks]

        # Johnson Algorithm
        j = johnson.JohnsonAlgorithm(sub_table)
        ordered_jobs = j.johnsons_algorithm()
        makespan, end_times, start_times = j.calculate_makespan(ordered_jobs)
        waiting_times = j.calculate_waiting_time(ordered_jobs, start_times)
        avg_waiting_time = j.calculate_average_waiting_time(waiting_times)
        response_times = j.calculate_response_time(ordered_jobs, end_times)
        avg_response_time = j.calculate_average_response_time(response_times)

        missed_deadlines, total_energy = calculate_missed_deadlines_and_energy(
            ordered_jobs, end_times, deadlines, base_energy, beta
        )

        data["Johnson"]["AvgWaitingTime1"].append(avg_waiting_time)
        data["Johnson"]["Waiting1"].append(waiting_times)
        data["Johnson"]["ResponseTime1"].append(response_times)
        data["Johnson"]["AvgResponseTime1"].append(avg_response_time)
        data["Johnson"]["Makespan1"].append(makespan)
        data["Johnson"]["MissedDeadlines1"].append(missed_deadlines)
        data["Johnson"]["TotalEnergy1"].append(total_energy)

        # Genetic Algorithm
        g = genetic.GeneticAlgorithm(sub_table)
        best_sequence = g.run()
        makespan, end_times, start_times = g.calculate_makespan(best_sequence)
        waiting_times = g.calculate_waiting_time(best_sequence, start_times)
        avg_waiting_time = g.calculate_average_waiting_time(waiting_times)
        response_times = g.calculate_response_time(best_sequence, end_times)
        avg_response_time = g.calculate_average_response_time(response_times)

        missed_deadlines, total_energy = calculate_missed_deadlines_and_energy(
            best_sequence, end_times, deadlines, base_energy, beta
        )

        data["Genetic"]["AvgWaitingTime1"].append(avg_waiting_time)
        data["Genetic"]["Waiting1"].append(waiting_times)
        data["Genetic"]["ResponseTime1"].append(response_times)
        data["Genetic"]["AvgResponseTime1"].append(avg_response_time)
        data["Genetic"]["Makespan1"].append(makespan)
        data["Genetic"]["MissedDeadlines1"].append(missed_deadlines)
        data["Genetic"]["TotalEnergy1"].append(total_energy)

        # DQN Algorithm
        env = JobshopEnvironment(sub_table, m, alpha, beta, W1, W2)
        agent = DQNAgent(state_size=env.state_size, action_size=env.action_size)

        # Pre-training using Johnson's algorithm
        train_dqn(agent, env, ordered_jobs, episodes=100, batch_size=32,
                  pretrain_steps=50)  # Include pretrain_steps

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

        dqn_makespan, dqn_end_times, dqn_start_times = j.calculate_makespan(env.schedule)

        dqn_waiting_times = env.calculate_waiting_time(dqn_start_times)
        dqn_avg_waiting_time = np.mean(dqn_waiting_times)
        dqn_response_times = env.calculate_response_time(dqn_end_times)
        dqn_avg_response_time = np.mean(dqn_response_times)

        missed_deadlines, total_energy = calculate_missed_deadlines_and_energy(
            sub_table, dqn_end_times, deadlines, base_energy, beta
        )

        data["DQN"]["AvgWaitingTime1"].append(dqn_avg_waiting_time)
        data["DQN"]["Waiting1"].append(dqn_waiting_times)
        data["DQN"]["ResponseTime1"].append(dqn_response_times)
        data["DQN"]["AvgResponseTime1"].append(dqn_avg_response_time)
        data["DQN"]["Makespan1"].append(dqn_makespan)
        data["DQN"]["MissedDeadlines1"].append(missed_deadlines)
        data["DQN"]["TotalEnergy1"].append(total_energy)

    # m constant , n variable
    for n in range(2, number_of_jobs + 1):
        sub_table = tasks[:n]
        sub_deadlines = deadlines[:n]
        sub_base_energy = base_energy[:n]

        # Johnson Algorithm
        j = johnson.JohnsonAlgorithm(sub_table)
        ordered_jobs = j.johnsons_algorithm()
        makespan, end_times, start_times = j.calculate_makespan(ordered_jobs)
        waiting_times = j.calculate_waiting_time(ordered_jobs, start_times)
        avg_waiting_time = j.calculate_average_waiting_time(waiting_times)
        response_times = j.calculate_response_time(ordered_jobs, end_times)
        avg_response_time = j.calculate_average_response_time(response_times)

        missed_deadlines, total_energy = calculate_missed_deadlines_and_energy(
            ordered_jobs, end_times, sub_deadlines, sub_base_energy, beta
        )

        data["Johnson"]["AvgWaitingTime2"].append(avg_waiting_time)
        data["Johnson"]["Waiting2"].append(waiting_times)
        data["Johnson"]["ResponseTime2"].append(response_times)
        data["Johnson"]["AvgResponseTime2"].append(avg_response_time)
        data["Johnson"]["Makespan2"].append(makespan)
        data["Johnson"]["MissedDeadlines2"].append(missed_deadlines)
        data["Johnson"]["TotalEnergy2"].append(total_energy)

        # Genetic Algorithm
        g = genetic.GeneticAlgorithm(sub_table)
        best_sequence = g.run()
        makespan, end_times, start_times = g.calculate_makespan(best_sequence)
        waiting_times = g.calculate_waiting_time(best_sequence, start_times)
        avg_waiting_time = g.calculate_average_waiting_time(waiting_times)
        response_times = g.calculate_response_time(best_sequence, end_times)
        avg_response_time = g.calculate_average_response_time(response_times)

        missed_deadlines, total_energy = calculate_missed_deadlines_and_energy(
            best_sequence, end_times, sub_deadlines, sub_base_energy, beta
        )

        data["Genetic"]["AvgWaitingTime2"].append(avg_waiting_time)
        data["Genetic"]["Waiting2"].append(waiting_times)
        data["Genetic"]["ResponseTime2"].append(response_times)
        data["Genetic"]["AvgResponseTime2"].append(avg_response_time)
        data["Genetic"]["Makespan2"].append(makespan)
        data["Genetic"]["MissedDeadlines2"].append(missed_deadlines)
        data["Genetic"]["TotalEnergy2"].append(total_energy)

        env = JobshopEnvironment(sub_table, number_of_machines, alpha, beta, W1, W2)
        agent = DQNAgent(state_size=env.state_size, action_size=env.action_size)

        # Pre-training using Johnson's algorithm
        train_dqn(agent, env, ordered_jobs, episodes=100, batch_size=32,
                  pretrain_steps=50)  # Include pretrain_steps

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

        dqn_makespan, dqn_end_times, dqn_start_times = j.calculate_makespan(env.schedule)

        dqn_waiting_times = env.calculate_waiting_time(dqn_start_times)
        dqn_avg_waiting_time = np.mean(dqn_waiting_times)
        dqn_response_times = env.calculate_response_time(dqn_end_times)
        dqn_avg_response_time = np.mean(dqn_response_times)

        missed_deadlines, total_energy = calculate_missed_deadlines_and_energy(
            sub_table, dqn_end_times, sub_deadlines, sub_base_energy, beta
        )

        data["DQN"]["AvgWaitingTime2"].append(dqn_avg_waiting_time)
        data["DQN"]["Waiting2"].append(dqn_waiting_times)
        data["DQN"]["ResponseTime2"].append(dqn_response_times)
        data["DQN"]["AvgResponseTime2"].append(dqn_avg_response_time)
        data["DQN"]["Makespan2"].append(dqn_makespan)
        data["DQN"]["MissedDeadlines2"].append(missed_deadlines)
        data["DQN"]["TotalEnergy2"].append(total_energy)

    pprint(data)
    helper_functions.plot_results(data, number_of_machines, number_of_jobs)


if __name__ == "__main__":
    main()
