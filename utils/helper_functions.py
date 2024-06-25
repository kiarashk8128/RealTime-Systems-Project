import numpy as np
import matplotlib.pyplot as plt
import pprint

def generate_samlpe_tasks(n, m):
    tasks = [(i, *[np.random.randint(1, 50) for _ in range(m)]) for i in range(n)]
    pprint.pprint(tasks)
    print("#############################################################")
    return tasks

def plot_results(data, number_of_machines, number_of_jobs):
    x_machines = range(2, number_of_machines + 1)
    x_jobs = range(2, number_of_jobs + 1)

    # Plot for Johnson Algorithm
    plt.figure(figsize=(10, 6))
    plt.plot(x_machines, data["Johnson"]["AvgResponseTime1"], marker='o',
             label="Johnson - Avg Response Time (m variable)")
    plt.xlabel("Number of Machines")
    plt.ylabel("Average Response Time")
    plt.title("Johnson Algorithm - Varying Machines (n constant)")
    plt.legend()
    plt.grid(True)

    plt.xticks(x_machines)

    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(x_jobs, data["Johnson"]["AvgResponseTime2"], marker='o', label="Johnson - Avg Response Time (n variable)")
    plt.xlabel("Number of Jobs")
    plt.ylabel("Average Response Time")
    plt.title("Johnson Algorithm - Varying Jobs (m constant)")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot for DQN Algorithm
    plt.figure(figsize=(10, 6))
    plt.plot(x_machines, data["DQN"]["AvgResponseTime1"], marker='o',
             label="DQN - Avg Response Time (m variable)")
    plt.xlabel("Number of Machines")
    plt.ylabel("Average Response Time")
    plt.title("DQN Algorithm - Varying Machines (n constant)")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(x_jobs, data["DQN"]["AvgResponseTime2"], marker='o', label="DQN - Avg Response Time (n variable)")
    plt.xlabel("Number of Jobs")
    plt.ylabel("Average Response Time")
    plt.title("DQN Algorithm - Varying Jobs (m constant)")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(x_machines, data["Johnson"]["AvgWaitingTime1"], marker='o',
             label="Johnson - Avg Waiting Time (m variable)")
    plt.xlabel("Number of Machines")
    plt.ylabel("Average Waiting Time")
    plt.title("Johnson Algorithm - Varying Machines (n constant)")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(x_jobs, data["Johnson"]["AvgWaitingTime2"], marker='o', label="Johnson - Avg Waiting Time (n variable)")
    plt.xlabel("Number of Jobs")
    plt.ylabel("Average Waiting Time")
    plt.title("Johnson Algorithm - Varying Jobs (m constant)")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(x_machines, data["DQN"]["AvgWaitingTime1"], marker='o',
             label="DQN - Avg Waiting Time (m variable)")
    plt.xlabel("Number of Machines")
    plt.ylabel("Average Waiting Time")
    plt.title("DQN Algorithm - Varying Machines (n constant)")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(x_jobs, data["DQN"]["AvgWaitingTime2"], marker='o', label="DQN - Avg Waiting Time (n variable)")
    plt.xlabel("Number of Jobs")
    plt.ylabel("Average Waiting Time")
    plt.title("DQN Algorithm - Varying Jobs (m constant)")
    plt.legend()
    plt.grid(True)
    plt.show()
