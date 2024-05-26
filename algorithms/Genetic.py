import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
from . import scheduler

class GeneticAlgorithm(scheduler.Scheduler):
    def __init__(self, jobs, population_size=100, generations=100, mutation_rate=0.01):
        self.jobs = jobs
        self.num_machines =  len(self.jobs[0])-1
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