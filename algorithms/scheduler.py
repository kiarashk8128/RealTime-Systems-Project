import numpy as np

class Scheduler:

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
