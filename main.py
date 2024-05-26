from algorithms import genetic
from algorithms import johnson
import pandas as pd
import numpy as np
from utils import helper_functions
import random
from pprint import pprint

number_of_machines = 5
number_of_jobs = 20

def main():
    tasks = helper_functions.generate_samlpe_tasks(number_of_jobs,number_of_machines)
    
    # delay1 & waiting1 --> refers to n constant and m variable
    # Delay2 & Waiting2 --> refers to m constant and n variable
    data = {
        "Johnson": {"Delay1": [], "Waiting1": [], "Delay2": [], "Waiting2": []},
        "Genetic": {"Delay1": [], "Waiting1": [], "Delay2": [], "Waiting2": []}
    }

# n constant , m variable 
    list1 = []
    for m in range(2,number_of_machines+1):
        sub_table = [row[:m+1] for row in tasks]
        j = johnson.JohnsonAlgorithm(sub_table)
        ordered_jobs = j.johnsons_algorithm()
        _, end_times = j.calculate_makespan(ordered_jobs)
    
        avg_delay = j.calculate_average_delay(end_times, ordered_jobs)
        avg_waiting_time = j.calculate_average_waiting_time(end_times)
        
        data["Johnson"]["Delay1"].append(avg_delay)
        data["Johnson"]["Waiting1"].append(avg_waiting_time)

######################
        continue
        g = genetic.GeneticAlgorithm(sub_table)
        best_sequence = g.run()
        _, end_times = g.calculate_makespan(best_sequence)
            
        avg_delay = g.calculate_average_delay(end_times, best_sequence)
        data["Genetic"]["Delay1"].append(avg_delay)
        avg_waiting_time = g.calculate_average_waiting_time(end_times)
        data["Genetic"]["Waiting1"].append(avg_waiting_time)

        list1.append([j, g])

    pprint(data)
    
# m constant , n variable
    list2 = []
    counter = 1
    for n in range(2, number_of_jobs+1):
        counter+=1
        print(counter)
        sub_table = tasks[:n]
        print(sub_table)
        # Johnson Algorithm
        j = johnson.JohnsonAlgorithm(sub_table)
        ordered_jobs = j.johnsons_algorithm()
        _, end_times = j.calculate_makespan(ordered_jobs)
        
        avg_delay = j.calculate_average_delay(end_times, ordered_jobs)
        avg_waiting_time = j.calculate_average_waiting_time(end_times)
        
        data["Johnson"]["Delay2"].append(avg_delay)
        data["Johnson"]["Waiting2"].append(avg_waiting_time)

        continue
        # Genetic Algorithm
        g = genetic.GeneticAlgorithm(sub_table)
        best_sequence = g.run()
        _, end_times = g.calculate_makespan(best_sequence)
        
        avg_delay = g.calculate_average_delay(end_times, best_sequence)
        avg_waiting_time = g.calculate_average_waiting_time(end_times)
        
        data["Genetic"]["Delay2"].append(avg_delay)
        data["Genetic"]["Waiting2"].append(avg_waiting_time)

        list2.append([j, g])
    # pass
    pprint(data)
    # helper_functions.plot_results(data, number_of_machines, number_of_jobs)
    

if __name__ == "__main__":
    main()
