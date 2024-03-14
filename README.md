# Knapsac Problem
## Table of contents
- [Introduction](#introduction)
- [How to use](#how-to-use)
    - [item_list.py](#item_list)
    - [main.py](#main)
- [Solved with](#solved-with)
    - [Genetic algorithm](#genetic-algorithm)
    - [Dynamic algorithm](#dynamic-algorithm)
    - [Greedy algorithm](#greedy-algorithm)
    - [Tabu search](#tabu-search)
    - [Bee algorithm](#bee-algorithm)
### Introduction
The program is made to solve common problem calles knapsac, where you have to put as many valuable items as possible into a backpack with limits. My solution is based on backpack ***max_weight*** and ***max_space*** and fitness function is the determinant on how good the solution is.
### How to use
#### item_list
In `/src` are two files one called `items_list.py` that generates database. To change the quantity of items change ***num_items***. Output file is calles `items_list.db` containing items with ***(id, weight, value, space)***, every single value is generated randomly.
#### main
The second file in `/src` is `main.py` that's the main core of the program containg every solution. Firstly it takes the database and read all the information then it's inserted into chosen algorithm *(more of them later)*. The algorithm is chosen by compiling the program and inserting appropriate value into terminal window:

<p align="center" width="100%">
    <img src="images/compiling.jpg">
</p>

There are some comments made in `main.py` for example:
<p align="center" width="100%">
    <img src="images/code.jpg">
</p>
First line shows where the variable is used, if it's not necessary for your algorithm just delete or comment it. Second and third are the algorithm and from which algorithm it's taking some functions.

### Solved with
#### Genetic algorithm
Genetic Algorithm (GA) works by iteratively evolving a population of candidate solutions towards optimal solutions through selection, crossover, and mutation operations. It mimics natural selection and genetic principles, evaluating and improving solutions based on their fitness until a termination criterion is met, yielding the best solution found.
#### Dynamis algorithm
Dynamic Programming (DP) breaks down a problem into smaller, overlapping subproblems, solving each one only once and storing the results to avoid redundant calculations. By iteratively solving these subproblems and building up to the larger problem, DP efficiently finds an optimal solution.
#### Greedy algorithm
Greedy Algorithm makes locally optimal choices at each step with the hope of finding a globally optimal solution. It iteratively selects the best available option at each stage without considering the overall problem, often leading to a suboptimal solution.
#### Tabu search
Tabu Search Algorithm explores the solution space iteratively by making moves based on a defined neighborhood structure while keeping track of previously visited solutions in a tabu list. It aims to escape local optima by forbidding certain moves for a limited number of iterations, promoting diversification, and ultimately converging to better solutions.
#### Bee algorithm
Bee Algorithm iteratively improves solutions by employing employed bees, onlooker bees, and scout bees. Employed bees explore local solutions, onlooker bees select promising solutions based on employed bees' information, and scout bees perform global exploration by discovering new solutions. This process balances exploration and exploitation to find high-quality solutions efficiently.
