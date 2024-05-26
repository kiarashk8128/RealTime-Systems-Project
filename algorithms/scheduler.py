import numpy as np

class Scheduler:

    def calculate_makespan(self, jobs):
        # Number of jobs
        num_jobs = len(jobs)
        # Number of machines
        num_machines = self.num_machines

        # Initialize a table to store the completion time of each job on each machine
        completion_times = np.zeros((num_jobs + 1, num_machines + 1)) # [[0] * (num_machines + 1) for _ in range(num_jobs + 1)]

        # Iterate through each job and machine to fill in the completion times
        for i in range(1, num_jobs + 1):
            _, *times = jobs[i - 1]
            for j in range(1, num_machines + 1):
                completion_times[i][j] = max(completion_times[i-1][j], completion_times[i][j-1]) + times[j-1]

        # The makespan is the completion time of the last job on the last machine
        makespan = completion_times[num_jobs][num_machines]
        return makespan, completion_times


    def calculate_average_delay(self, end_times, jobs):
        return np.mean([end_times[i][-1] - sum(jobs[i][1:]) for i in range(len(jobs))])


    def calculate_average_waiting_time(self, end_times):
        return np.mean(end_times[:, 1:] - end_times[:, :-1])