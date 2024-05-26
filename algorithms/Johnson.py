import matplotlib.pyplot as plt
from . import scheduler
import numpy as np

class JohnsonAlgorithm(scheduler.Scheduler):
    def __init__(self, jobs):
        self.jobs = jobs
        self.num_machines = len(self.jobs[0])-1
    
    def johnsons_algorithm(self, adhoc=False):
        if adhoc:
            return self.johnsons_algorithm_multiple_machines_adhoc(self.jobs, self.num_machines)
        if self.num_machines == 2:
            return self.johnsons_algorithm_two_machines(self.jobs)
        return self.johnsons_algorithm_multiple_machines(self.jobs, self.num_machines)
    
    def johnsons_algorithm_two_machines(self, jobs):
        # Sorting tasks based on the minimum processing time on either machine
        # Task format: (id, time_machine_1, time_machine_2)
        sorted_tasks = sorted(jobs, key=lambda x: min(x[1], x[2]))

        front, back = [], []
        for task in sorted_tasks:
            if task[1] < task[2]:
                front.append(task)
            else:
                back.append(task)
        
        # Front list is in normal order, back list is reversed
        return front + back[::-1]

    def johnsons_algorithm_multiple_machines(self, jobs, num_machines):
        # Reduce multiple machines to two pseudo machines
        pseudo_jobs = [
            (job[0], sum(job[1:num_machines//2+1]), sum(job[num_machines//2+1:])) 
            for job in jobs
        ]
        ordered_jobs = self.johnsons_algorithm_two_machines(pseudo_jobs)
        
        # Map back to original jobs
        job_map = {job[0]: job for job in jobs}
        ordered_jobs = [job_map[job[0]] for job in ordered_jobs]
        
        return ordered_jobs
    
    def johnsons_algorithm_multiple_machines_adhoc(self, jobs, num_machines):
        if num_machines == 2:
            return self.johnsons_algorithm_two_machines(jobs)
        
        # Reduce multiple machines to two pseudo machines
        job_map = {job[0]: job for job in jobs}
        pseudo_jobs = [
            (job[0], np.average(job[1:num_machines//2+1]), np.average(job[num_machines//2+1:]))
            for job in jobs
        ]
        ordered_jobs = np.array(self.johnsons_algorithm_two_machines(pseudo_jobs), dtype=int)
        
        for index in range(len(ordered_jobs)):
            if ordered_jobs[index, 2] <= ordered_jobs[index, 1]:
                left_part = np.array(ordered_jobs[:index, 0])
                if num_machines > 3:
                    left_part = np.array(self.johnsons_algorithm_multiple_machines_adhoc([job_map[i][:num_machines//2+1] for i in ordered_jobs[:index, 0]], num_machines//2))[:, 0]
                right_part = np.array(self.johnsons_algorithm_multiple_machines_adhoc([(i,) + job_map[i][num_machines//2+1:] for i in ordered_jobs[index:, 0]], num_machines - num_machines//2))[:, 0]

                # Map back to original jobs
                final_order = list(left_part) + list(right_part[::-1])
                return [job_map[i] for i in final_order]
        
        # Map back to original jobs
        ordered_jobs = [job_map[job[0]] for job in ordered_jobs]
        return ordered_jobs