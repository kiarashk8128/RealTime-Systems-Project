# Real-Time Systems Project

## Overview
This repository focuses on real-time systems, specifically tackling the problem of Job-Shop Scheduling. The project implements various algorithms such as Johnson's algorithm (extended for more than two machines), genetic algorithms, and Deep Q-Learning to address scheduling challenges.

## Features
- **Algorithms**: Implements Johnson's algorithm (extended version), genetic algorithms, and Deep Q-Learning for scheduling.
- **Main Components**: 
  - `dqn.py`: Deep Q-Learning implementation.
  - `main.py`: Main script to run the project.
  - `algorithms/`: Contains specific algorithms for scheduling.
  - `utils/`: Utility functions used across the project.

## Algorithms

### Johnson's Algorithm
Johnson's algorithm is traditionally used for two-machine job shops to minimize makespan. In this project, the algorithm has been extended to handle more than two machines, providing a more robust solution for complex scheduling tasks.

### Genetic Algorithms
Genetic algorithms (GAs) are optimization techniques inspired by natural selection. They are particularly useful for solving complex scheduling problems. The GA in this project uses selection, crossover, and mutation operations to evolve solutions over generations, aiming to find an optimal or near-optimal job schedule.

### Deep Q-Learning
Deep Q-Learning is a reinforcement learning algorithm that combines Q-Learning with deep neural networks. It is used to learn optimal scheduling policies by approximating the Q-value function, which helps in making decisions that maximize cumulative rewards. This approach is particularly powerful for dynamic and complex scheduling environments.

## Getting Started
1. Clone the repository:
   ```sh
   git clone https://github.com/Ardalan-Sia/RealTime-Systems-Project.git
   ```
2. Navigate to the project directory:
   ```sh
   cd RealTime-Systems-Project
   ```
3. Install required dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## Usage
Run the main script to execute the scheduling algorithms:
```sh
python main.py
```

## Contributors
- Ardalan-Sia
- Kiarash Kianian
