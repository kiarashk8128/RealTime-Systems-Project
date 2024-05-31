import numpy as np

class Scheduler:

    def calculate_makespan(self, jobs):
        # Number of jobs
        num_jobs = len(jobs)
        # Number of machines
        self.num_machines = len(jobs[0]) - 1

        # Initialize a table to store the completion time of each job on each machine
        completion_times = np.zeros((num_jobs + 1, self.num_machines + 1))

        # Initialize a table to store the start time of each job on each machine
        start_times = np.zeros((num_jobs + 1, self.num_machines + 1))

        # Iterate through each job and machine to fill in the completion times
        for i in range(1, num_jobs + 1):
            job_id, *times = jobs[i - 1]
            for j in range(1, self.num_machines + 1):
                start_time = max(completion_times[i - 1][j], completion_times[i][j - 1])
                start_times[i][j] = start_time
                completion_times[i][j] = start_time + times[j - 1]

        # The makespan is the completion time of the last job on the last machine
        makespan = completion_times[num_jobs][self.num_machines]
        return makespan, completion_times, start_times

    def calculate_response_time(self, jobs, completion_times):
        num_jobs = len(jobs)
        response_times = [completion_times[i][-1] for i in range(1, num_jobs + 1)]
        return response_times

    def calculate_average_response_time(self, response_times):
        return np.mean(response_times)

    def calculate_waiting_time(self, jobs, start_times):
        num_jobs = len(jobs)
        waiting_times = [start_times[i][1] for i in range(1, num_jobs + 1)]
        return waiting_times

    def calculate_average_waiting_time(self, waiting_times):
        return np.mean(waiting_times)
