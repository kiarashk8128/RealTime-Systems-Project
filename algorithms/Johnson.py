import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

class JohnsonAlgorithm:
    def __init__(self, jobs):
        self.jobs = jobs
        self.num_machines = len(self.jobs[0])
    
    def johnsons_algorithm(self):
        if self.num_machines == 2:
            return self.johnsons_algorithm_two_machines(self.jobs)
        else:
            return self.johnsons_algorithm_multiple_machines(self.jobs, self.num_machines)
    
    def johnsons_algorithm_two_machines(self, jobs):
        group1 = []
        group2 = []
        
        for job in jobs:
            if job[1] < job[2]:
                group1.append(job)
            else:
                group2.append(job)
        
        group1.sort(key=lambda x: x[1])
        group2.sort(key=lambda x: x[2], reverse=True)
        
        return group1 + group2

    def johnsons_algorithm_multiple_machines(self, jobs, num_machines):
        # Reduce multiple machines to two pseudo machines
        pseudo_jobs = [(job[0], sum(job[1:num_machines//2 + 1]), sum(job[num_machines//2 + 1:])) for job in jobs]
        ordered_jobs = self.johnsons_algorithm_two_machines(pseudo_jobs)
        
        # Map back to original jobs
        job_map = {job[0]: job for job in jobs}
        ordered_jobs = [job_map[job[0]] for job in ordered_jobs]
        
        return ordered_jobs

    def calculate_makespan(self, jobs):
        end_times = np.zeros((len(jobs), self.num_machines))
        
        for i, job in enumerate(jobs):
            for j in range(1, self.num_machines + 1):
                if i == 0 and j == 1:
                    end_times[i][j-1] = job[j]
                elif i == 0:
                    end_times[i][j-1] = end_times[i][j-2] + job[j]
                elif j == 1:
                    end_times[i][j-1] = end_times[i-1][j-1] + job[j]
                else:
                    end_times[i][j-1] = max(end_times[i-1][j-1], end_times[i][j-2]) + job[j]
        
        return end_times[-1][-1], end_times

    def calculate_average_delay(self, end_times, jobs):
        total_delay = 0
        for i, job in enumerate(jobs):
            total_delay += end_times[i][-1] - sum(job[1:self.num_machines+1])
        return total_delay / len(jobs)

    def calculate_average_waiting_time(self, end_times):
        waiting_times = end_times[:, :-1] - end_times[:, 1:]
        return np.mean(waiting_times)

    def plot_chart(self, x_values, y_values, title, x_label, y_label):
        plt.figure(figsize=(10, 6))
        plt.plot(x_values, y_values, marker='o')
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.grid(True)
        plt.show()

    def generate_charts(self):
        delays = []
        waiting_times = []
        n_values = range(5, 30, 5)
        
        for n in n_values:
            sample_jobs = self.jobs[:n]
            ordered_jobs = self.johnsons_algorithm()
            _, end_times = self.calculate_makespan(ordered_jobs)
            
            avg_delay = self.calculate_average_delay(end_times, ordered_jobs)
            avg_waiting_time = self.calculate_average_waiting_time(end_times)
            
            delays.append(avg_delay)
            waiting_times.append(avg_waiting_time)
        
        self.plot_chart(n_values, delays, "Average Delay with varying n", "Number of Jobs (n)", "Average Delay")
        self.plot_chart(n_values, waiting_times, "Average Waiting Time with varying n", "Number of Jobs (n)", "Average Waiting Time")

# Example usage
johnson = JohnsonAlgorithm('task_samples.csv')
johnson.generate_charts()
