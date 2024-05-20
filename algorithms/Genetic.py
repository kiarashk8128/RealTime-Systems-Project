import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random


class GeneticAlgorithm:
    def __init__(self, jobs, population_size=100, generations=1000, mutation_rate=0.01):
        self.jobs = jobs
        self.num_machines = len(self.jobs_df.columns) - 1
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate

    def initialize_population(self):
        population = []
        for _ in range(self.population_size):
            individual = random.sample(self.jobs, len(self.jobs))
            population.append(individual)
        return population
    
    def fitness(self, individual):
        _, end_times = self.calculate_makespan(individual)
        return end_times[-1][-1]
    
    def selection(self, population):
        # Tournament selection
        selected = random.sample(population, 2)
        return min(selected, key=self.fitness)
    
    def crossover(self, parent1, parent2):
        # Order Crossover (OX)
        size = len(parent1)
        start, end = sorted(random.sample(range(size), 2))
        
        offspring = [None] * size
        offspring[start:end] = parent1[start:end]
        
        for job in parent2:
            if job not in offspring:
                for i in range(size):
                    if offspring[i] is None:
                        offspring[i] = job
                        break
        return offspring
    
    def mutate(self, individual):
        if random.random() < self.mutation_rate:
            i, j = random.sample(range(len(individual)), 2)
            individual[i], individual[j] = individual[j], individual[i]
        return individual
    
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

    def run(self):
        population = self.initialize_population()
        
        for _ in range(self.generations):
            population = sorted(population, key=self.fitness)
            new_population = population[:self.population_size//2]
            
            while len(new_population) < self.population_size:
                parent1 = self.selection(population)
                parent2 = self.selection(population)
                offspring = self.crossover(parent1, parent2)
                offspring = self.mutate(offspring)
                new_population.append(offspring)
            
            population = new_population
        
        best_solution = min(population, key=self.fitness)
        return best_solution

# charts
    def generate_charts(self):
        delays = []
        waiting_times = []
        n_values = range(5, 30, 5)
        
        for n in n_values:
            sample_jobs = self.jobs[:n]
            best_sequence = self.run()
            _, end_times = self.calculate_makespan(best_sequence)
            
            avg_delay = self.calculate_average_delay(end_times, best_sequence)
            avg_waiting_time = self.calculate_average_waiting_time(end_times)
            
            delays.append(avg_delay)
            waiting_times.append(avg_waiting_time)
        
        self.plot_chart(n_values, delays, "Average Delay with varying n", "Number of Jobs (n)", "Average Delay")
        self.plot_chart(n_values, waiting_times, "Average Waiting Time with varying n", "Number of Jobs (n)", "Average Waiting Time")
    
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

# Example usage
genetic = GeneticAlgorithm('task_samples.csv')
genetic.generate_charts()
