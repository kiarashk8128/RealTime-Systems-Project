import matplotlib.pyplot as plt
import algorithms.Genetic
import algorithms.Johnson
import pandas as pd
import numpy as np
import random

def generate_task_samples(file_path='task_samples.csv', num_jobs=20, num_machines=5):
    samples = []
    for job in range(1, num_jobs + 1):
        job_id = f'Job{job}'
        processing_times = [random.randint(1, 20) for _ in range(num_machines)]
        samples.append([job_id] + processing_times)
    
    columns = ['Job'] + [f'Machine{i+1}' for i in range(num_machines)]
    df = pd.DataFrame(samples, columns=columns)
    df.to_csv(file_path, index=False)
    print(f'Task samples saved to {file_path}')

# Example usage
generate_task_samples(num_jobs=20, num_machines=5)
string_path = "data.csv"

def main():
# m constant , n variable for avarage delay
# n constant , m variable for avarage waiting time
# m constant , n variable for average delay
# m constant , n variable for average waition time
    pass


if __name__ == "__main__":
    pass
