# 1. Завантажити біблітеку PyGad
# 2. Визначити окремо в коді фітнес функцію.
# 3. Визначити початкову популяцію.
# 4. Задати параметри генетичного алгоритму: кількість популяцій, кількість батьківських хромосом, що приймають участь у кросовері, тип кросовера та мутації тощо.
# 5. Створити екземпляр генеричного алгоритму засобами PyGad.
# 6. Дослідити вплив різних кросоверів та мутацій на результат.
# 7. Зробити висновок про ефективність застосування генетичного алгоритму та оптимальний набір його параметрів.


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import pygad

DIV = "-" * 200
MAX_TRACK_SPACE = 3.0  #  (м³)


class Product:
    def __init__(self, name, space, price):
        self.name = name
        self.space = space
        self.price = price


def populate_product_list():
    inventory_items = []
    inventory_items.append(Product("Refrigerator A", 0.751, 999.90))
    inventory_items.append(Product("Cell phone", 0.00000899, 2199.12))
    inventory_items.append(Product("TV 55'", 0.400, 4346.99))
    inventory_items.append(Product("TV 50'", 0.290, 3999.90))
    inventory_items.append(Product("TV 42'", 0.200, 2999.00))
    inventory_items.append(Product("Notebook A", 0.00350, 2499.90))
    inventory_items.append(Product("Ventilator", 0.496, 199.90))
    inventory_items.append(Product("Microwave A", 0.0424, 308.66))
    inventory_items.append(Product("Microwave B", 0.0544, 429.90))
    inventory_items.append(Product("Microwave C", 0.0319, 299.29))
    inventory_items.append(Product("Refrigerator B", 0.635, 849.00))
    inventory_items.append(Product("Refrigerator C", 0.870, 1199.89))
    inventory_items.append(Product("Notebook B", 0.498, 1999.90))
    inventory_items.append(Product("Notebook C", 0.527, 3999.00))

    return inventory_items


def print_product_list(products_list):

    print("Products list:")
    print("\n", DIV, "\n", f'{"Name":<20} {"Space":<10} {"Price":<10}', "\n", DIV)
    for product in products_list:
        print(f"{product.name:<20} {product.space:<10} {product.price:<10}")
    print(DIV)


def fitness_function(ga_instance, solution, solution_idx):

    products = ga_instance.products_list

    spaces = np.array([product.space for product in products])
    prices = np.array([product.price for product in products])

    total_space = np.sum(solution * spaces)
    total_price = np.sum(solution * prices)

    if total_space > MAX_TRACK_SPACE:
        return 0  # Invalid solution, space constraint violated
    else:
        return total_price  # Valid solution, return total price as fitness


def optimize_products_with_ga(fitness_function, products_list, crossovers, mutations, initial_population):

    optimization_results = []

    for crossover_type in crossovers:
        for mutation_type in mutations:
            ga = pygad.GA(
                initial_population=initial_population,
                fitness_func=fitness_function,
                num_genes=len(products_list),
                crossover_type=crossover_type,
                mutation_type=mutation_type,
                gene_type=int,  # Ensure genes are integers (0 or 1)
                gene_space=[0, 1],  # Restrict gene values to 0 or 1  (1 when product is taken, 0 when product is not taken)
                num_generations=200, # кількість поколінь для еволюції
                num_parents_mating=6, # кількість батьківських хромосом, що приймають участь у кросовері
                parent_selection_type="tournament",  # "sss" - Steady State Selection, "rws" - Roulette Wheel Selection, "rank" - Rank Selection, "random" - Random Selection, "tournament" - Tournament Selection
                mutation_percent_genes=10,  # Відсоток генів для мутації
                
                # "saturate_10" - зупиняє алгоритм, якщо найкраще рішення не покращується протягом 10 поколінь, 
                # "reach_10000" - зупиняє алгоритм, якщо досягається фітнес 10000, 
                # "stop_on_low_fitness" - зупиняє алгоритм, якщо фітнес падає нижче певного порогу, 
                # "num_generations" - зупиняє алгоритм після заданої кількості поколінь
                stop_criteria=["reach_24281", # результат тестових запусків
                               #"saturate_50" # зупиняє алгоритм, якщо найкраще рішення не покращується протягом 50 поколінь
                ]
            )

            ga.products_list = products_list
            ga.run()
            solution, solution_fitness, _ = ga.best_solution()

            total_space = np.sum(np.array(solution) * np.array([product.space for product in products_list]))
            selected_products = [products_list[i].name for i, g in enumerate(solution) if int(round(g)) == 1]

            optimization_results.append((crossover_type, mutation_type, ga.generations_completed, solution, solution_fitness, total_space, selected_products))

    return optimization_results


if __name__ == "__main__":
    products_list = populate_product_list()
    print_product_list(products_list)

    product_spaces = np.array([product.space for product in products_list])
    product_prices = np.array([product.price for product in products_list])

    np.random.seed(42)  # Answer to the Ultimate Question of Life, the Universe, and Everything
    POPULATION_SIZE = 20  # кількість особин у популяції

    # 0 when product is not taken,
    # 1 when product is taken
    initial_population = np.random.randint(low=0, high=2, size=(POPULATION_SIZE, len(products_list))).tolist()

    print(f"\nInitial population {POPULATION_SIZE} chromosomes by {len(products_list)} genes:")
    print("\n", DIV, "\n", f'{"Index":<10} {"Chromosome":<20} {"Space":<15} {"Price":<15} {"Valid":<10}', "\n", DIV)
    for i, chromosome in enumerate(initial_population):
        total_s = np.sum(np.array(chromosome) * product_spaces)
        total_p = np.sum(np.array(chromosome) * product_prices)
        valid = total_s <= MAX_TRACK_SPACE
        print(f"{i:<10} {str(chromosome):<20} {total_s:<15.4f} {total_p:<15.2f} {str(valid):<10}")
    print(DIV)

    crossover_types = ["single_point", "two_points", "uniform", "scattered"]
    mutation_types = ["random", "swap", "inversion", "scramble"]
    results = optimize_products_with_ga(fitness_function, products_list, crossover_types, mutation_types, initial_population)
    
    print("\nOptimization results:")

    df = pd.DataFrame(results, columns=["Crossover Type", "Mutation Type", "Generations", "Best Solution", "Best Fitness", "Total Space", "Selected Products"])
    print(df.to_string(index=False))

    print("Genetic Algorithm Summary:")
    print("- Best fitness achieved: 24281.55")
    print("- Setting stop criteria to 'reach_24281' allowed to decrease unnecessary computations.")
    print("- With stop criteria to 'reach_24281' it's enough to achieve optimal solution in less than 50 generations in most cases, significantly reducing computational time.")
    print("- Optimal solution found consistently across most configurations.")    
    print("- Mutation type has major impact with current configuration:")
    print("  * Best: random, inversion, scramble")
    print("  * Worst: swap (often failed to converge)")
    
    