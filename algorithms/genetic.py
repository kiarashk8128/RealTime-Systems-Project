import random
import numpy as np
import queue
from . import scheduler



class GeneticAlgorithm(scheduler.Scheduler):
    def __init__(self, jobs, population_size=100, generations=100, mutation_rate=0.01):
        self.jobs = jobs
        self.num_machines = len(self.jobs[0]) - 1
        self.num_tasks = len(self.jobs)
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.population = self.initialize_population()

    def initialize_population(self):
        population = []
        for _ in range(self.population_size):
            individual = list(np.random.permutation(self.num_tasks))
            population.append(individual)
        return population
    
    def calculateObj(self, sol):
        qTime = queue.PriorityQueue()
        qMachines = [queue.Queue() for _ in range(self.num_machines)]
        for i in range(self.num_tasks):
            qMachines[0].put(sol[i])
        
        busyMachines = [False] * self.num_machines
        time = 0
        
        task_id = qMachines[0].get()
        qTime.put((time + self.jobs[task_id][1], 0, task_id))
        busyMachines[0] = True
        
        while True:
            time, mach, task_id = qTime.get()
            if task_id == sol[self.num_tasks - 1] and mach == self.num_machines - 1:
                break
            busyMachines[mach] = False
            if not qMachines[mach].empty():
                next_task_id = qMachines[mach].get()
                qTime.put((time + self.jobs[next_task_id][mach + 1], mach, next_task_id))
                busyMachines[mach] = True
            if mach < self.num_machines - 1:
                if not busyMachines[mach + 1]:
                    qTime.put((time + self.jobs[task_id][mach + 2], mach + 1, task_id))
                    busyMachines[mach + 1] = True
                else:
                    qMachines[mach + 1].put(task_id)
                    
        return time

    def fitness(self, individual):
        return self.calculateObj(individual)
    
    def selection(self, population):
        selected = random.sample(population, 2)
        return min(selected, key=self.fitness)
    
    def crossover(self, parent1, parent2):
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

    def run(self):
        population = self.population
        
        for _ in range(self.generations):
            population = sorted(population, key=self.fitness)
            new_population = population[:self.population_size // 2]
            
            while len(new_population) < self.population_size:
                parent1 = self.selection(population)
                parent2 = self.selection(population)
                offspring = self.crossover(parent1, parent2)
                offspring = self.mutate(offspring)
                new_population.append(offspring)
            
            population = new_population
        
        best_solution_indices = min(population, key=self.fitness)
        best_solution = [self.jobs[i] for i in best_solution_indices]
        return best_solution


