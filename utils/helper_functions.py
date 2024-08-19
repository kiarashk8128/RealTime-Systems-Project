# gernerate smaple tasks
# claculate delay
# calculate waiting time
# ...
import numpy as np
import matplotlib.pyplot as plt
import pprint
import random


def generate_real_data_sample_tasks(n, m):
    # Hardcoded data from the provided table
    benchmark_data = {
        'BITCOUNT': {'time': 193.15, 'energy': 112.21},
        'SUSAN': {'time': 118.09, 'energy': 67.95},
        'MATH': {'time': 1098.40, 'energy': 604.26},
        'CRC32': {'time': 2078.51, 'energy': 1107.95},
        'SHA': {'time': 39.36, 'energy': 22.51},
        'QSORT': {'time': 206.82, 'energy': 120.18},
        'JPEG': {'time': 47.89, 'energy': 29.44},
        'FFT': {'time': 960.88, 'energy': 554.07},
        'DIJKSTRA': {'time': 89.90, 'energy': 56.59},
        'LAME': {'time': 3055.44, 'energy': 1925.32},
        'GSM': {'time': 704.46, 'energy': 409.51}
    }

    benchmark_keys = list(benchmark_data.keys())
    selected_operations = {key: 0 for key in benchmark_keys}

    tasks = []
    energy_consumptions = []
    for i in range(n):
        task_times = [i]  # Start with the job_id
        energy_sum = 0
        for _ in range(m):
            selected_key = random.choice(benchmark_keys)
            selected_operations[selected_key] += 1
            task_times.append(int(benchmark_data[selected_key]['time']))  # Convert to int for simplicity
            energy_sum += benchmark_data[selected_key]['energy']
        tasks.append(tuple(task_times))  # Convert list to tuple
        energy_consumptions.append(energy_sum)

    pprint.pprint(tasks)
    print("#############################################################")
    return tasks, energy_consumptions, selected_operations



def generate_samlpe_tasks(n, m):
    tasks = [(i, *[np.random.randint(1, 50) for _ in range(m)]) for i in range(n)]
    pprint.pprint(tasks)
    print("#############################################################")
    return tasks


def calculate_missed_deadlines_and_energy(tasks, end_times, deadlines, base_energy, beta):
    missed_deadlines = 0
    total_energy = 0.0

    for i, (job, deadline, energy) in enumerate(zip(tasks, deadlines, base_energy)):
        finish_time = end_times[i][-1]  # Finish time of the job on the last machine
        if finish_time > deadline:
            missed_deadlines += 1
            extra_energy = (finish_time - deadline) * beta
            total_energy += energy + extra_energy
        else:
            total_energy += energy

    return missed_deadlines, total_energy

def plot_combined_makespan_energy(data, x_values, x_label, title_suffix, makespan_key, energy_key):
    plt.figure(figsize=(10, 6))

    # Bar plot for Makespan
    plt.bar(x_values, data["Johnson"][makespan_key], color='blue', alpha=0.5, label="Johnson - Makespan")
    plt.bar(x_values, data["Genetic"][makespan_key], color='green', alpha=0.5, label="Genetic - Makespan")
    plt.bar(x_values, data["DQN"][makespan_key], color='red', alpha=0.5, label="DQN - Makespan")

    # Line plot for Energy Consumption
    plt.plot(x_values, data["Johnson"][energy_key], color='blue', marker='o', linestyle='-', label="Johnson - Energy")
    plt.plot(x_values, data["Genetic"][energy_key], color='green', marker='o', linestyle='-', label="Genetic - Energy")
    plt.plot(x_values, data["DQN"][energy_key], color='red', marker='o', linestyle='-', label="DQN - Energy")

    plt.xlabel(x_label)
    plt.ylabel("Makespan / Total Energy")
    plt.title(f"Makespan and Total Energy - {title_suffix}")
    plt.legend()
    plt.grid(True)
    plt.show()




def plot_results(data, number_of_machines, number_of_jobs):
    x_machines = range(2, number_of_machines + 1)
    x_jobs = range(2, number_of_jobs + 1)

    # Plot for Average Response Time - Varying Machines
    plt.figure(figsize=(10, 6))
    plt.plot(x_machines, data["Johnson"]["AvgResponseTime1"], marker='o',
             label="Johnson - Avg Response Time (m variable)")
    plt.plot(x_machines, data["Genetic"]["AvgResponseTime1"], marker='o',
             label="Genetic - Avg Response Time (m variable)")
    plt.plot(x_machines, data["DQN"]["AvgResponseTime1"], marker='o',
             label="DQN - Avg Response Time (m variable)")
    plt.xlabel("Number of Machines")
    plt.ylabel("Average Response Time")
    plt.title("Average Response Time - Varying Machines (n constant)")
    plt.legend()
    plt.grid(True)
    plt.xticks(x_machines)
    plt.show()

    # Plot for Average Response Time - Varying Jobs
    plt.figure(figsize=(10, 6))
    plt.plot(x_jobs, data["Johnson"]["AvgResponseTime2"], marker='o', label="Johnson - Avg Response Time (n variable)")
    plt.plot(x_jobs, data["Genetic"]["AvgResponseTime2"], marker='o', label="Genetic - Avg Response Time (n variable)")
    plt.plot(x_jobs, data["DQN"]["AvgResponseTime2"], marker='o', label="DQN - Avg Response Time (n variable)")
    plt.xlabel("Number of Jobs")
    plt.ylabel("Average Response Time")
    plt.title("Average Response Time - Varying Jobs (m constant)")
    plt.legend()
    plt.grid(True)
    plt.xticks(x_jobs)
    plt.show()

    # Plot for Average Waiting Time - Varying Machines
    plt.figure(figsize=(10, 6))
    plt.plot(x_machines, data["Johnson"]["AvgWaitingTime1"], marker='o',
             label="Johnson - Avg Waiting Time (m variable)")
    plt.plot(x_machines, data["Genetic"]["AvgWaitingTime1"], marker='o',
             label="Genetic - Avg Waiting Time (m variable)")
    plt.plot(x_machines, data["DQN"]["AvgWaitingTime1"], marker='o',
             label="DQN - Avg Waiting Time (m variable)")
    plt.xlabel("Number of Machines")
    plt.ylabel("Average Waiting Time")
    plt.title("Average Waiting Time - Varying Machines (n constant)")
    plt.legend()
    plt.grid(True)
    plt.xticks(x_machines)
    plt.show()

    # Plot for Average Waiting Time - Varying Jobs
    plt.figure(figsize=(10, 6))
    plt.plot(x_jobs, data["Johnson"]["AvgWaitingTime2"], marker='o', label="Johnson - Avg Waiting Time (n variable)")
    plt.plot(x_jobs, data["Genetic"]["AvgWaitingTime2"], marker='o', label="Genetic - Avg Waiting Time (n variable)")
    plt.plot(x_jobs, data["DQN"]["AvgWaitingTime2"], marker='o', label="DQN - Avg Waiting Time (n variable)")
    plt.xlabel("Number of Jobs")
    plt.ylabel("Average Waiting Time")
    plt.title("Average Waiting Time - Varying Jobs (m constant)")
    plt.legend()
    plt.grid(True)
    plt.xticks(x_jobs)
    plt.show()

    # Plot for Missed Deadlines - Varying Machines
    plt.figure(figsize=(10, 6))
    plt.plot(x_machines, data["Johnson"]["MissedDeadlines1"], marker='o',
             label="Johnson - Missed Deadlines (m variable)")
    plt.plot(x_machines, data["Genetic"]["MissedDeadlines1"], marker='o',
             label="Genetic - Missed Deadlines (m variable)")
    plt.plot(x_machines, data["DQN"]["MissedDeadlines1"], marker='o',
             label="DQN - Missed Deadlines (m variable)")
    plt.xlabel("Number of Machines")
    plt.ylabel("Missed Deadlines")
    plt.title("Missed Deadlines - Varying Machines (n constant)")
    plt.legend()
    plt.grid(True)
    plt.xticks(x_machines)
    plt.show()

    # Plot for Missed Deadlines - Varying Jobs
    plt.figure(figsize=(10, 6))
    plt.plot(x_jobs, data["Johnson"]["MissedDeadlines2"], marker='o',
             label="Johnson - Missed Deadlines (n variable)")
    plt.plot(x_jobs, data["Genetic"]["MissedDeadlines2"], marker='o',
             label="Genetic - Missed Deadlines (n variable)")
    plt.plot(x_jobs, data["DQN"]["MissedDeadlines2"], marker='o',
             label="DQN - Missed Deadlines (n variable)")
    plt.xlabel("Number of Jobs")
    plt.ylabel("Missed Deadlines")
    plt.title("Missed Deadlines - Varying Jobs (m constant)")
    plt.legend()
    plt.grid(True)
    plt.xticks(x_jobs)
    plt.show()

    # Plot for Total Energy Consumption - Varying Machines
    plt.figure(figsize=(10, 6))
    plt.plot(x_machines, data["Johnson"]["TotalEnergy1"], marker='o',
             label="Johnson - Total Energy (m variable)")
    plt.plot(x_machines, data["Genetic"]["TotalEnergy1"], marker='o',
             label="Genetic - Total Energy (m variable)")
    plt.plot(x_machines, data["DQN"]["TotalEnergy1"], marker='o',
             label="DQN - Total Energy (m variable)")
    plt.xlabel("Number of Machines")
    plt.ylabel("Total Energy")
    plt.title("Total Energy - Varying Machines (n constant)")
    plt.legend()
    plt.grid(True)
    plt.xticks(x_machines)
    plt.show()

    # Plot for Total Energy Consumption - Varying Jobs
    plt.figure(figsize=(10, 6))
    plt.plot(x_jobs, data["Johnson"]["TotalEnergy2"], marker='o', label="Johnson - Total Energy (n variable)")
    plt.plot(x_jobs, data["Genetic"]["TotalEnergy2"], marker='o', label="Genetic - Total Energy (n variable)")
    plt.plot(x_jobs, data["DQN"]["TotalEnergy2"], marker='o', label="DQN - Total Energy (n variable)")
    plt.xlabel("Number of Jobs")
    plt.ylabel("Total Energy")
    plt.title("Total Energy - Varying Jobs (m constant)")
    plt.legend()
    plt.grid(True)
    plt.xticks(x_jobs)
    plt.show()

    # Plot for Makespan - Varying Machines
    plt.figure(figsize=(10, 6))
    plt.plot(x_machines, data["Johnson"]["Makespan1"], marker='o',
             label="Johnson - Makespan (m variable)")
    plt.plot(x_machines, data["Genetic"]["Makespan1"], marker='o',
             label="Genetic - Makespan (m variable)")
    plt.plot(x_machines, data["DQN"]["Makespan1"], marker='o',
             label="DQN - Makespan (m variable)")
    plt.xlabel("Number of Machines")
    plt.ylabel("Makespan")
    plt.title("Makespan - Varying Machines (n constant)")
    plt.legend()
    plt.grid(True)
    plt.xticks(x_machines)
    plt.show()

    # Plot for Makespan - Varying Jobs
    plt.figure(figsize=(10, 6))
    plt.plot(x_jobs, data["Johnson"]["Makespan2"], marker='o', label="Johnson - Makespan (n variable)")
    plt.plot(x_jobs, data["Genetic"]["Makespan2"], marker='o', label="Genetic - Makespan (n variable)")
    plt.plot(x_jobs, data["DQN"]["Makespan2"], marker='o', label="DQN - Makespan (n variable)")
    plt.xlabel("Number of Jobs")
    plt.ylabel("Makespan")
    plt.title("Makespan - Varying Jobs (m constant)")
    plt.legend()
    plt.grid(True)
    plt.xticks(x_jobs)
    plt.show()

    plot_combined_makespan_energy(data, x_machines, "Number of Machines", "Varying Machines (n constant)", "Makespan1", "TotalEnergy1")


    # Combined Plot for Varying Jobs
    plot_combined_makespan_energy(data, x_jobs, "Number of Jobs", "Varying Jobs (m constant)", "Makespan2", "TotalEnergy2")

