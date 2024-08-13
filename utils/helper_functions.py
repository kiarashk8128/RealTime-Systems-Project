# gernerate smaple tasks
# claculate delay
# calculate waiting time
# ...
import numpy as np
import matplotlib.pyplot as plt
import pprint


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
