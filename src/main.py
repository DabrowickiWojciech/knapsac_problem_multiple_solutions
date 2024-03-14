import numpy as np
import sqlite3

''' Variables '''
# Define knapsack problem parameters
max_weight = 50
max_space = 80
population_size = 50 # genetic, tabu
# Always best answer at 5000, 99 centile at 100 
generations = 100 # genetic
mutation_rate = 0.1 # genetic, tabu
tabu_size = 10 # tabu
iterations = 100 # tabu 100, bee 100
bee_count = 50 # bee
# Number of times a solution can be revisited before it's abandoned
limit = 3 # bee

''' Connectiong to the data base '''
# Connect to the database
conn = sqlite3.connect('items_list.db')
c = conn.cursor()

# Fetch items from the database
c.execute("SELECT weight, value, space FROM items")
items_data = c.fetchall()
conn.close()

''' linking item data  '''
# Unpack items_data into weights and values lists
items_data = sorted(items_data, key=lambda x: x[1] / x[0], reverse=True) # This line is required for tabu, branch algorithm
weights, values, spaces = zip(*items_data)

''' Functions for genetic algorithm '''
# Initialize population randomly
def initialize_population(population_size):
    # Initialize a population of size population_size with binary representation of items
    population = np.random.randint(2, size=(population_size, len(weights)))
    return population.tolist()

# Calculate fitness of each individual
def calculate_fitness(individual):
    # Calculate the total weight, total space, and total value of the knapsack for an individual
    total_weight = np.sum(np.array(weights) * np.array(individual))
    total_space = np.sum(np.array(spaces) * np.array(individual))
    total_value = np.sum(np.array(values) * np.array(individual))
    # Check if the individual violates the weight or space constraints
    if total_weight > max_weight or total_space > max_space:
        return 0  # Return 0 fitness if violated
    else:
        return total_value  # Otherwise, return the total value as fitness

# Selection - Best Two Parents
def selection(population, fitness_scores):
    # Sort the population based on fitness scores in descending order
    sorted_population = [x for _, x in sorted(zip(fitness_scores, population), reverse=True)]
    # Select the best two individuals as parents
    parent1 = sorted_population[0]
    parent2 = sorted_population[1]
    return parent1, parent2

# Crossover - Single Point Crossover
def crossover(parent1, parent2):
    # Randomly select a crossover point
    crossover_point = np.random.randint(len(parent1))
    # Perform single-point crossover
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2

# Mutation
def mutation(individual):
    # Perform mutation on individual with a probability mutation_rate
    for i in range(len(individual)):
        if np.random.rand() < mutation_rate:
            individual[i] = 1 - individual[i]  # Flip the bit
    return individual

# Genetic Algorithm
def genetic_algorithm():
    # Initialize variables to keep track of the best fitness and generation
    best_fitness = 0
    best_generation = 0
    
    # Initialize the population
    population = initialize_population(population_size)
    
    # Loop through generations
    for generation in range(generations):
        # Calculate fitness scores for each individual in the population
        fitness_scores = [calculate_fitness(individual) for individual in population]
        
        # Find the best fitness score in the current generation
        current_best_fitness = max(fitness_scores)
        
        # Update the best fitness and generation if a better fitness is found
        if current_best_fitness > best_fitness:
            best_fitness = current_best_fitness
            best_generation = generation
            
        # Select the best individual from the current population
        best_individual_index = np.argmax(fitness_scores)
        new_population = [population[best_individual_index]]
        
        # Selection, crossover, and mutation
        selected_population = selection(population, fitness_scores)
        for i in range(int(population_size / 2)):
            parent1, parent2 = selected_population
            child1, child2 = crossover(parent1, parent2)
            new_population.extend([child1, child2])
        mutated_population = [mutation(individual) for individual in new_population]
        population = mutated_population
        
        # Print generation and best fitness for monitoring
        print("Generation:", generation, "Best Fitness:", current_best_fitness)
        
    # Return the best overall fitness and the generation where it was achieved
    return best_fitness, best_generation

''' Dynamic algorithm '''
# Function to solve knapsack problem using dynamic programming
def knapsack_dynamic(weights, values, max_weight, max_space):
    n = len(weights)
    dp = [[0] * (max_space + 1) for _ in range(max_weight + 1)]

    for i in range(1, n + 1):
        for j in range(max_weight, -1, -1):
            for k in range(max_space, -1, -1):
                if j >= weights[i - 1] and k >= 1:
                    dp[j][k] = max(dp[j][k], dp[j - weights[i - 1]][k - 1] + values[i - 1])

    return dp[max_weight][max_space]

''' Greedy algorithm '''
# Function to solve knapsack problem using a greedy algorithm
def knapsack_greedy(weights, values, max_weight, max_space):
    n = len(weights)
    total_weight = 0
    total_space = 0
    max_value = 0

    for i in range(n):
        if total_weight + weights[i] <= max_weight and total_space + 1 <= max_space:
            total_weight += weights[i]
            total_space += 1
            max_value += values[i]
    
    return max_value
        
'''Tabu search''' 
# From genethic:  calculate_fitness(), initialize_population()
# Define a move operator to generate neighboring solutions
def move_operator(individual):
    index = np.random.randint(0, len(individual) - 1)
    neighbor = individual[:]
    neighbor[index] = 1 - neighbor[index]  # Flip bit
    return neighbor

# Define the tabu search algorithm
def tabu_search():
    # Initialize the current solution and best solution
    current_solution = initialize_population(1)[0]
    best_solution = current_solution[:]
    best_fitness = calculate_fitness(best_solution)

    # Initialize tabu list
    tabu_list = []

    # Perform iterations
    for _ in range(iterations):
        neighbors = [move_operator(current_solution) for _ in range(population_size)]

        # Select the best non-tabu neighbor
        best_neighbor = None
        best_neighbor_fitness = float('-inf')
        for neighbor in neighbors:
            neighbor_fitness = calculate_fitness(neighbor)
            if neighbor_fitness > best_neighbor_fitness and neighbor not in tabu_list:
                best_neighbor = neighbor
                best_neighbor_fitness = neighbor_fitness

        # Update current solution
        current_solution = best_neighbor

        # Update tabu list
        tabu_list.append(current_solution)
        if len(tabu_list) > tabu_size:
            tabu_list.pop(0)

        # Update best solution if applicable
        if best_neighbor_fitness > best_fitness:
            best_solution = best_neighbor
            best_fitness = best_neighbor_fitness
            
        # Print generation and best fitness for monitoring
        print("Generation:", iterations, "Best Fitness:", current_solution)

    return best_solution, best_fitness

''' Branch and bound algorithm'''
# Function to calculate upper bound of a node
def calculate_upper_bound(weight, value, index, remaining_weight, remaining_space):
    upper_bound = value
    total_weight = weight
    total_space = 1
    while index < len(weights) and total_weight + weights[index] <= remaining_weight and total_space <= remaining_space:
        upper_bound += values[index]
        total_weight += weights[index]
        total_space += 1
        index += 1
    if index < len(weights) and total_space <= remaining_space:
        upper_bound += (remaining_weight - total_weight) * (values[index] / weights[index])
    return upper_bound

# Define the node structure for branch and bound
class Node:
    def __init__(self, index, weight, value, path, remaining_weight, remaining_space, upper_bound):
        self.index = index
        self.weight = weight
        self.value = value
        self.path = path
        self.remaining_weight = remaining_weight
        self.remaining_space = remaining_space
        self.upper_bound = upper_bound

# Branch and bound algorithm
def branch_and_bound():
    best_value = 0
    best_path = []

    initial_node = Node(0, 0, 0, [], max_weight, max_space, 0)
    priority_queue = [initial_node]

    while priority_queue:
        node = priority_queue.pop(0)

        if node.index == len(weights):
            if node.value > best_value:
                best_value = node.value
                best_path = node.path
            continue

        include_node = Node(
            node.index + 1,
            node.weight + weights[node.index],
            node.value + values[node.index],
            node.path + [1],
            node.remaining_weight - weights[node.index],
            node.remaining_space - 1,
            0
        )

        if include_node.weight <= max_weight and include_node.remaining_space >= 0:
            include_node.upper_bound = calculate_upper_bound(include_node.weight, include_node.value, include_node.index, include_node.remaining_weight, include_node.remaining_space)
            if include_node.upper_bound > best_value:
                priority_queue.append(include_node)

        exclude_node = Node(
            node.index + 1,
            node.weight,
            node.value,
            node.path + [0],
            node.remaining_weight,
            node.remaining_space,
            0
        )

        exclude_node.upper_bound = calculate_upper_bound(exclude_node.weight, exclude_node.value, exclude_node.index, exclude_node.remaining_weight, exclude_node.remaining_space)
        if exclude_node.upper_bound > best_value:
            priority_queue.append(exclude_node)

        priority_queue.sort(key=lambda x: -x.upper_bound)

    return best_value, best_path

''' Bee algorythm '''
# Function to calculate fitness of a solution
def calculate_fitness(individual):
    total_weight = np.sum(np.array(weights) * np.array(individual))
    total_space = np.sum(individual)
    total_value = np.sum(np.array(values) * np.array(individual))
    if total_weight > max_weight or total_space > max_space:
        return 0
    else:
        return total_value

# Function to generate initial solutions
def generate_solutions():
    return [np.random.randint(2, size=len(weights)) for _ in range(bee_count)]

# Function to perform local search on a solution
def local_search(solution):
    best_solution = solution.copy()
    best_fitness = calculate_fitness(best_solution)

    for _ in range(5):  # Number of local search iterations
        # Randomly select two indices
        i, j = np.random.choice(range(len(solution)), size=2, replace=False)
        # Swap items at indices i and j
        new_solution = best_solution.copy()
        new_solution[i], new_solution[j] = new_solution[j], new_solution[i]
        # Calculate fitness of new solution
        new_fitness = calculate_fitness(new_solution)
        # Update best solution if new solution is better
        if new_fitness > best_fitness:
            best_solution = new_solution
            best_fitness = new_fitness

    return best_solution

# Bee Algorithm
def bee_algorithm():
    best_solution = None
    best_fitness = 0
    abandon_count = np.zeros(bee_count)

    solutions = generate_solutions()

    for _ in range(iterations):
        for i, solution in enumerate(solutions):
            local_solution = local_search(solution)
            fitness = calculate_fitness(local_solution)
            if fitness > best_fitness:
                best_solution = local_solution
                best_fitness = fitness
                abandon_count = np.zeros(bee_count)
            elif abandon_count[i] < limit:
                solutions[i] = local_solution
                abandon_count[i] += 1
            else:
                solutions[i] = np.random.randint(2, size=len(weights))
                abandon_count[i] = 0

    return best_solution, best_fitness


''' Main loop '''
def main():
    option = 1
    ''' Selecting algorithm '''
    option = int(input("Select the algorithm you want to use \n(1) - Genetic algorithm\n(2) - Dynamic algorithm\n(3) - Greedy algorithm\n(4) - Tabu search algorithm\n(5) - Branch and bound algorithm\n(6) - Bee algorithm\nInput: "))
    
    if option == 1:
        # Run the genetic algorithm
        best_fitness, best_generation = genetic_algorithm()

        # Print the best overall fitness and the generation where it was achieved
        print("Best overall fitness:", best_fitness)
        print("Achieved at generation:", best_generation)
    elif option == 2:
        # Solve knapsack problem using dynamic programming
        max_value = knapsack_dynamic(weights, values, max_weight, max_space)

        # Output the result
        print("Maximum value that can be obtained:", max_value)
    elif option == 3:
        # Solve knapsack problem using greedy algorithm
        max_value_greedy = knapsack_greedy(weights, values, max_weight, max_space)

        # Output the result
        print("Maximum value that can be obtained using greedy algorithm:", max_value_greedy)    
    elif option == 4:
        # Run tabu search algorithm
        best_solution, best_fitness = tabu_search()

        # Output the best solution and its fitness
        print("Best solution:", best_solution)
        print("Best fitness:", best_fitness)    
    elif option == 5:
        # Solve knapsack problem using Branch and Bound algorithm
        best_value, best_path = branch_and_bound()

        # Output the result
        print("Best value that can be obtained using branch and bound:", best_value)
        print("Items selected:", best_path)
    elif option == 6:    
        # Run Bee Algorithm
        best_solution, best_fitness = bee_algorithm()

        # Output the best solution and its fitness
        print("Best solution:", best_solution)
        print("Best fitness:", best_fitness)
    else:
        print("Wrong input !")
        return 0       
            
if __name__ == "__main__":
    main()