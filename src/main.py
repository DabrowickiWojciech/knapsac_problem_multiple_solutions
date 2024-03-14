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
items_data = sorted(items_data, key=lambda x: x[1] / x[0], reverse=True) # This line is required for tabu
weights, values, spaces = zip(*items_data)

''' Functions for genetic algorithm '''
# Initialize population randomly
def initialize_population(population_size):
    # Initialize a population of size population_size with binary representation of items
    population = np.random.randint(2, size=(population_size, len(values)))
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
        for _ in range(int(population_size / 2)):
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
def knapsack_dynamic(weights, values, spaces, max_weight, max_space):
    n = len(values)
    dp = [[0] * (max_space + 1) for _ in range(max_weight + 1)]

    for i in range(1, n + 1):
        for j in range(max_weight, -1, -1):
            for k in range(max_space, -1, -1):
                if j >= weights[i - 1] and k >= spaces[i - 1] and k >= 1:
                    dp[j][k] = max(dp[j][k], dp[j - weights[i - 1]][k - spaces[i - 1]]+ values[i - 1])

    return dp[max_weight][max_space]

''' Greedy algorithm '''
# Function to solve knapsack problem using a greedy algorithm
def knapsack_greedy(weights, values, spaces, max_weight, max_space):
    n = len(weights)
    total_weight = 0
    total_space = 0
    max_value = 0

    for i in range(n):
        if total_weight + weights[i] <= max_weight and total_space + spaces[i] <= max_space:
            total_weight += weights[i]
            total_space += spaces[i]
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
    for i in range(iterations):
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
        print("Generation:", i, "Best Fitness:", current_solution)

    return best_solution, best_fitness

''' Bee algorythm '''
#From genethic:  calculate_fitness(), initialize_population()
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

    solutions = initialize_population(population_size)

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
            print("Generation:", i, "Best Fitness:", local_solution)

    return best_solution, best_fitness


''' Main loop '''
def main():
    option = 1
    ''' Selecting algorithm '''
    option = int(input("Select the algorithm you want to use \n(1) - Genetic algorithm\n(2) - Dynamic algorithm\n(3) - Greedy algorithm\n(4) - Tabu search algorithm\n(5) - Bee algorithm\nInput: "))
    
    if option == 1:
        # Run the genetic algorithm
        best_fitness, best_generation = genetic_algorithm()

        # Print the best overall fitness and the generation where it was achieved
        print("Best overall fitness:", best_fitness)
        print("Achieved at generation:", best_generation)
    elif option == 2:
        # Solve knapsack problem using dynamic programming
        max_value = knapsack_dynamic(weights, values, spaces, max_weight, max_space)

        # Output the result
        print("Maximum value that can be obtained:", max_value)
    elif option == 3:
        # Solve knapsack problem using greedy algorithm
        max_value_greedy = knapsack_greedy(weights, values, spaces, max_weight, max_space)

        # Output the result
        print("Maximum value that can be obtained using greedy algorithm:", max_value_greedy)    
    elif option == 4:
        # Run tabu search algorithm
        best_solution, best_fitness = tabu_search()

        # Output the best solution and its fitness
        print("Best solution:", best_solution)
        print("Best fitness:", best_fitness)    
    elif option == 5:    
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