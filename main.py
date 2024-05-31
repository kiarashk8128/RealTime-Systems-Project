from algorithms import genetic
from algorithms import johnson
import pandas as pd
import numpy as np
from utils import helper_functions
import random
from pprint import pprint

number_of_machines = 2
number_of_jobs = 5


def main():
    tasks = helper_functions.generate_samlpe_tasks(number_of_jobs, number_of_machines)
    print(tasks)

    # delay1 & waiting1 --> refers to n constant and m variable
    # Delay2 & Waiting2 --> refers to m constant and n variable
    data = {
        "Johnson": {"AvgWaitingTime1": [], "Waiting1": [], "AvgWaitingTime2": [], "Waiting2": [], "ResponseTime1": [],
                    "ResponseTime2": [], "AvgResponseTime1": [], "AvgResponseTime2": []},
        "Genetic": {"Delay1": [], "Waiting1": [], "Delay2": [], "Waiting2": [], "ResponseTime1": [],
                    "ResponseTime2": [], "AvgResponseTime1": [], "AvgResponseTime2": []}
    }

    # n constant , m variable
    list1 = []
    for m in range(2, number_of_machines + 1):
        sub_table = [row[:m + 1] for row in tasks]
        j = johnson.JohnsonAlgorithm(sub_table)
        ordered_jobs = j.johnsons_algorithm()
        makespan, end_times, start_times = j.calculate_makespan(ordered_jobs)
        waiting_times = j.calculate_waiting_time(ordered_jobs, start_times)
        avg_waiting_time = j.calculate_average_waiting_time(waiting_times)
        response_times = j.calculate_response_time(ordered_jobs, end_times)
        avg_response_time = j.calculate_average_response_time(response_times)

        data["Johnson"]["AvgWaitingTime1"].append(avg_waiting_time)
        data["Johnson"]["Waiting1"].append(waiting_times)
        data["Johnson"]["ResponseTime1"].append(response_times)
        data["Johnson"]["AvgResponseTime1"].append(avg_response_time)


        # continue
        # g = genetic.GeneticAlgorithm(sub_table)
        # best_sequence = g.run()
        # makespan, end_times, start_times = g.calculate_makespan(best_sequence)
        #
        # avg_delay = g.calculate_average_delay(end_times, best_sequence)
        # avg_waiting_time = g.calculate_average_waiting_time(end_times)
        # response_times = g.calculate_response_time(best_sequence, end_times)
        # avg_response_time = g.calculate_average_response_time(response_times)
        #
        # data["Genetic"]["Delay1"].append(avg_delay)
        # data["Genetic"]["Waiting1"].append(avg_waiting_time)
        # data["Genetic"]["ResponseTime1"].append(response_times)
        # data["Genetic"]["AvgResponseTime1"].append(avg_response_time)
        #
        # list1.append([j, g])

    pprint(data)

    # m constant , n variable
    list2 = []
    counter = 1
    for n in range(2, number_of_jobs + 1):
        counter += 1
        print(counter)
        sub_table = tasks[:n]
        print(sub_table)
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

        # continue
        # # Genetic Algorithm
        # g = genetic.GeneticAlgorithm(sub_table)
        # best_sequence = g.run()
        # makespan, end_times, start_times = g.calculate_makespan(best_sequence)
        #
        # avg_delay = g.calculate_average_delay(end_times, best_sequence)
        # avg_waiting_time = g.calculate_average_waiting_time(end_times)
        # response_times = g.calculate_response_time(best_sequence, end_times)
        # avg_response_time = g.calculate_average_response_time(response_times)
        #
        # data["Genetic"]["Delay2"].append(avg_delay)
        # data["Genetic"]["Waiting2"].append(avg_waiting_time)
        # data["Genetic"]["ResponseTime2"].append(response_times)
        # data["Genetic"]["AvgResponseTime2"].append(avg_response_time)
        #
        # list2.append([j, g])

    pprint(data)
    # helper_functions.plot_results(data, number_of_machines, number_of_jobs)


if __name__ == "__main__":
    main()