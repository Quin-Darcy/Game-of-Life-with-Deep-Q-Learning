# Deep Q-Learning for Game of Life

## Overview
This project leverages Deep Q-Learning (DQN) to optimize initial grid states in Conway's Game of Life, aiming to maximize population age, diversity, and growth. The program features an intricate blend of reinforcement learning, genetic algorithms, and neural network training, culminating in the visualization of learned grid states using the Nannou framework.

## Features
- **Reinforcement Learning Agent**: Employs exploration and exploitation strategies for generating and evolving grid states.
- **Genetic Algorithm**: Integrates a genetic algorithm for state space evolution, featuring tournament selection, crossover, mutation, and population pruning.
- **Deep Q-Network (DQN)**: Utilizes a DQN to learn the optimal toggling actions for grid cells, enhancing the quality of the grid states.
- **Visualization with Nannou**: Implements the Nannou framework for visualizing the progression and outcome of the learned grid states.

## Technical Workflow
1. **Initialization**: The program initializes a DQN structure alongside a reinforcement learning agent.
2. **Agent Interaction Phase**:
   - **Exploration**: The agent generates random grid states, with a probability influenced by the performance metrics of the highest-performing grids in its population.
   - **Exploitation**: Employs a genetic algorithm that evolves the state space. The algorithm includes tournament selection (selecting about 70% as winners), crossover (swapping random-sized rectangular sections between two parent states), mutation (flipping random cells based on a mutation rate), and population pruning to maintain a manageable state space size.
   - **State Evaluation**: Grid states are evaluated based on population age, standard deviation of population size over time, and overall population growth. 
3. **Replay Buffer Formation**: Converts the agent's state space into a set of experiences, forming the replay buffer.
4. **Neural Network Training**:
   - The replay buffer is segmented into batches.
   - Each epoch processes the batches to compute average weight/bias gradients, subsequently updating the main network's parameters.
   - Synchronization of the target network with the main network occurs post each epoch.
5. **Result Generation and Visualization**: Generates a random grid state, iteratively modified based on the network's max Q-value predictions, visualized through hundreds of iterations proportional to the grid size using Nannou.

## Implementation Details
- **Actions Definition**: Actions are defined as toggling individual cells, equating the number of possible actions to the grid size.
- **Cyclic Average and Population Control**: Includes cyclic average monitoring to avoid simple repetitive structures, terminating grid evaluation once a population size repetition threshold is exceeded.
- **State Space and Genetic Algorithm**: The agent's state space is dynamically evolved using a genetic algorithm, balancing between exploration of new states and exploitation of high-performing existing states.

## Usage
- Instructions on building and running the program.
- Dependency and prerequisite information.

## TODO
- Add feedback logic so that states iteratively produced by the trained network are added back into the Agent's state space and subsequently, the replay buffer.
- Write more unit tests
- Add more error handling
- Use multithreading for the bottlenecks

## Contributing
- Guidelines for those interested in contributing to the project.

## License
- License details for the project.
