from algorithms import genetic
from algorithms import johnson
import pandas as pd
import numpy as np
from utils import helper_functions
import random



number_of_machines = 5
number_of_jobs = 20

def main():
    tasks = helper_functions.generate_samlpe_tasks(number_of_jobs,number_of_machines)
    
    # delay1 & waiting1 --> refers to n constant and m variable
    # delay2 & waiting2 --> refers to m constant and n variable
    data = {
        "Johnson": {"delay1": [], "waiting1": [],"delay2": [], "waiting2": []},
        "Genetic": {"delay1": [], "waiting1": [],"delay2": [], "waiting2": []}
    }

# n constant , m variable 
    list1 = []
    for m in range(0,number_of_machines) :
        sub_table = [row[:m] for row in tasks]
        j = johnson.JohnsonAlgorithm(sub_table)
        ordered_jobs = j.johnsons_algorithm()
        _, end_times = j.calculate_makespan(ordered_jobs)
    
        avg_delay = j.calculate_average_delay(end_times, ordered_jobs)
        avg_waiting_time = j.calculate_average_waiting_time(end_times)
        
        data["Johnson"]["delay2"].append(avg_delay)
        data["Johnson"]["waiting2"].append(avg_waiting_time)
######################
        g = genetic.GeneticAlgorithm(sub_table)
        best_sequence = g.run()
        _, end_times = g.calculate_makespan(best_sequence)
            
        avg_delay = g.calculate_average_delay(end_times, best_sequence)
        data["Genetic"]["delay2"].append(avg_delay)
        avg_waiting_time = g.calculate_average_waiting_time(end_times)
        data["Genetic"]["waiting2"].append(avg_waiting_time)

        list1.append([j, g])


# m constant , n variable
    list2 = []
    for n in range(0,number_of_jobs) :
        sub_table = tasks[:n]
        j = johnson.JohnsonAlgorithm(sub_table)
        g = genetic.GeneticAlgorithm(sub_table)
        list2.append([j, g])
    pass


if __name__ == "__main__":
    pass
