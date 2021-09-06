

from algorithms.aoa import AOA

import numpy as np
import time
import matplotlib.pyplot as plt

from utilities.function import *

function = Rastrigin(20)

aoa = AOA(50,function)

aoa = AOA(50,function)

aoa.initial_solutions()

C_iter = 1
M_iter = 100
fitness = np.arange(M_iter+1)

#print(fitness)

objective_aoa = np.zeros(M_iter+1)
fitness_aoa = np.zeros(M_iter+1)

#print(fitness_fa)

start_time_aoa = time.time()

#print(start_time_fa)

while C_iter < M_iter:
    aoa.update_position(M_iter)
    aoa.sort_population()
    objective_aoa[C_iter] = aoa.get_global_best()
    fitness_aoa[C_iter] = aoa.get_global_best_fitness()
    C_iter = C_iter + 1

total_time_aoa = time.time() - start_time_aoa

global_best_aoa = aoa.get_global_best()
global_worst_aoa = aoa.get_global_worst()
median_aoa = aoa.median_result()
average_result_aoa = aoa.average_result()
std_aoa = aoa.std_result()
optimizer_aoa = aoa.algorithm()

# Print results
print('Results of {}: \n'.format(optimizer_aoa))
print('Time: {:.2f} seconds '.format(total_time_aoa))
print('Global best: {:.20f} '.format(global_best_aoa))
print('Global worst: {:.6f} '.format(global_worst_aoa))
print('Median: {:.6f} '.format(median_aoa))
print('Average: {:.6f} '.format(average_result_aoa))
print('Std: {:.6f} '.format(std_aoa))
print()

aoa.optimum()

#print('\nBest solutions: ')
#fa.print_global_parameters()

plt.plot(fitness_aoa)
plt.title("Fitness convergence")
plt.show()

plt.plot(objective_aoa)
plt.title("Objective convergence")
plt.show()


aoa.print_global_parameters()
