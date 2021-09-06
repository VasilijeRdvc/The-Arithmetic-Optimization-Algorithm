import numpy as np

class Solution:
    def __init__(self, function, x=None):
        if x is None:
            self.function = function
            self.x = [None] * self.function.D  
            self.objective_function = None
            self.fitness = None
            self.initialize()
            self.trial = 0
        else:
            self.x = x
            self.function = function
            self.calculate_objective_function()
            self.calculate_fitness()
            self.trial = 0
      

    def initialize(self):        
        for i in range(0, len(self.x)):
            rnd = np.random.rand()
            self.x[i] = rnd * (self.function.ub[i] - self.function.lb[i]) + self.function.lb[i]

        self.calculate_objective_function()
        self.calculate_fitness()
        

    def calculate_objective_function(self):
        self.objective_function = self.function.function(self.x)
        self.fitness = 1 / (1 + abs(self.objective_function))

    def calculate_fitness(self):
        self.fitness = 1 / (1 + abs(self.objective_function))
        

