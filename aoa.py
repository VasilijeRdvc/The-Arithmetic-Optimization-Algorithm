import math
from utilities.solution import Solution
import random
import copy
import numpy as np


class AOA:
    def __init__(self, n, function):

        self.N = n
        self.function = function
        self.population = []
        self.best_solution = [None] * self.function.D
        self.u = 0.5
        self.alpha = 5
        self.MOA_min = 0.2
        self.MOA_max = 0.9
        self.C_iter = 1;

        self.best_position = np.zeros(self.function.D)
        self.best_score = float("inf")
        

    def initial_solutions(self):
        for i in range(1, self.N):
            local_solution = Solution(self.function)
            self.population.append(local_solution)

        self.population.sort(key=lambda x: x.fitness)
        self.best_solution = copy.deepcopy(self.population[-1].x)

    def update_position(self, M_iter):
        while self.C_iter < M_iter:
        
            for i in range(1, self.N):
                fitness = self.population[-1].fitness
                Xbest = copy.deepcopy(self.population[-1].x)

                if fitness > self.best_score:
                    self.best_score = fitness
                    self.best_position = Xbest

            MOP = 1 - ((self.C_iter)**(1/self.alpha)/(M_iter)**(1/self.alpha))
            MOA = self.MOA_min + self.C_iter * ((self.MOA_max - self.MOA_min) / M_iter)

            for i in range(self.N):
                Xbest = copy.deepcopy(self.population[-1].x)
                Xnew = [None] * self.function.D
                
                for j in range(self.function.D):
                    r1 = np.random.rand()
                    r2 = np.random.rand()
                    r3 = np.random.rand()

                    UB = self.function.ub[j]
                    LB = self.function.lb[j]

                    if r1 > MOA:

                        #Exploration phase

                        if r2 > 0.5: # < ?
                            # Division
                            Xnew[j] = Xbest[j] / (MOP + math.e) * ((UB - LB) * self.u + LB)

                        else:
                            # Multiplication
                            Xnew[j] = Xbest[j] * MOP * ((UB - LB) * self.u + LB)

                    else:

                        # Exploitation phase

                        if r3 > 0.5: # < ?
                            # Subtraction
                            Xnew[j] = Xbest[j] - MOP * ((UB - LB) * self.u + LB)

                        else:
                            # Addiction
                            Xnew[j] = Xbest[j] + MOP * ((UB - LB) * self.u + LB)

                    if Xnew[j] < LB:
                        Xnew[j] = LB
                    
                    elif Xnew[j] > UB:
                        Xnew[j] = UB

                self.solution = Solution(self.function, Xnew)

                if self.solution.fitness > self.population[-1].fitness:
                    self.population[i-1] = self.solution

            self.population.sort(key=lambda x: x.fitness)
            self.best_solution = copy.deepcopy(self.population[-1].x)

            self.C_iter = self.C_iter +1


        return self.best_solution


    def sort_population(self):

        self.population.sort(key=lambda x: x.fitness)
        self.best_solution = self.population[0].x

    def get_global_best(self):
        return self.population[-1].objective_function
    
    def get_global_worst(self):
        return self.population[0].objective_function

    def get_global_best_fitness(self):
        return self.population[-1].fitness

    def optimum(self):
        print('f(x*) = ', self.function.minimum, 'at x* = ', self.function.solution)
        
    def algorithm(self):
        return 'AOA'

    def objective(self):
        
        result = []
        
        for i in range(self.N):
            result.append(self.population[-1].objective_function)
            
        return result
    
    def average_result(self):
        return np.mean(np.array(self.objective()))
    
    def std_result(self):        
        return np.std(np.array(self.objective()))
    
    def median_result(self):
        return np.median(np.array(self.objective()))
    
    
    def print_global_parameters(self):
            for i in range(0, len(self.best_solution)):
                 print('X: {}'.format(self.best_solution[i]))
                 
    def get_best_solutions(self):
        return np.array(self.best_solution)

    def get_solutions(self):
        
        sol = np.zeros((self.N, self.function.D))
        for i in range(len(self.population)):
            sol[i] = np.array(self.population[-1].x)
        return sol


    def print_all_solutions(self):
        print("******all solutions objectives**********")
        for i in range(0,len(self.population)):

              print('solution {}'.format(i))
              print('objective:{}'.format(self.population[i].objective_function))
              print('fitness:{}'.format(self.population[i].fitness))
              print('solution {}: '.format(self.population[i].x))
              print('--------------------------------------')










