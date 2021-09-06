#original
#from algorithms.abc_algorithm import ABCAlgorithm
#from algorithms.basicABC import BasicABC
#from algorithms.bat import BAT
#from algorithms.eho import EHO
#from algorithms.fa import FA
#from algorithms.hho import HHO
#from algorithms.mbo import MBO
#from algorithms.ssa import SSA
#from algorithms.woa import WOA

# hybrid
#from algorithms.abc_gbest_min_max import ABC_GBEST
#from algorithms.abc_qrbest import ABC_QRBEST
#from algorithms.abc_qrbest_Nebojsa import ABC_QRBEST_N
#from algorithms.gbestBasicABC import GBestBasicABC
#from algorithms.gbestMinMaxQRABC import GbestMinMaxQRABC
#from algorithms.GOQRFA import GOQRFA
#from algorithms.mfa import mFA
#from algorithms.modifiedABC import ModifiedABC


from algorithms.aoa import AOA
from utilities.function import *

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



def func(function, dimension):
    
    if function == 'Sphere':
        x = Sphere(dimension)
        f = 'F1'
    elif function == 'MovedAxisFunction':
        x = MovedAxisFunction(dimension)
        f = 'F2'
    elif function == 'Griewank':
        x = Griewank(dimension)
        f = 'F3'
    elif function == 'Rastrigin':
        x = Rastrigin(dimension)
        f = 'F4'
    elif function == 'Schwefel_1_2':
        x = Schwefel_1_2(dimension)
        f = 'F5'
    elif function == 'Ackley':
        x = Ackley(dimension)
        f = 'F6'
    elif function == 'PowellSum':
        x = PowellSum(dimension)
        f = 'F7'
    elif function == 'SumSquares':
        x = SumSquares(dimension)
        f = 'F8'
    elif function == 'Schwefel_2_22':
        x = Schwefel_2_22(dimension)
        f = 'F9'
    elif function == 'PowellSingular':
        x = PowellSingular(dimension)
        f = 'F10'      
    elif function == 'Alpine':
        x = Alpine(dimension)
        f = 'F11'
    elif function == 'InvertedCosineWaveFunction':
        x = InvertedCosineWaveFunction(dimension)
        f = 'F12'
    elif function == 'Pathological':
        x = Pathological(dimension)
        f = 'F13'
    elif function == 'Discus':
        x = Discus(dimension)
        f = 'F14'
    elif function == 'HappyCat':
        x = HappyCat(dimension)
        f = 'F15' 
    elif function == 'DropWaveFunction':
        x = DropWaveFunction(dimension)
        f = 'F16'
    elif function == 'Schaffer2':
        x = Schaffer2(dimension)
        f = 'F17'
    elif function == 'CamelBackThreeHump':
        x = CamelBackThreeHump(dimension)
        f = 'F18' 
    return x, f


def opt(optimizer, population_size, function):
    
    if optimizer == 'AOA':
        x = AOA(population_size, function)
    #elif optimizer == 'HHO':
     #   x = FA(population_size, function)
    ## TO DO
    return x

optimizer_list = ["AOA"]
function_list = ["Sphere", "MovedAxisFunction", "Griewank", "Rastrigin", "Schwefel_1_2", 
                 "Ackley", "PowellSum", "SumSquares", "Schwefel_2_22", "PowellSingular",
                 "Alpine", "InvertedCosineWaveFunction", "Pathological", "Discus" , "HappyCat"]


#function_list = ["DropWaveFunction","Schaffer2", "CamelBackThreeHump"]

dim = [5,10,15]



# the population size of the six algorithms is 20, 
# the maximum number of iterations MaxG = 1000
# 50 independent runs

population_size = 10
num_run = 5
M_iter = 200



for h in range (len(dim)):
    optimizer_data = []
    dimension_data = []
    function_data = []
    function_id_data = []
    run_data =  []
    time_data = []
    iteration_data = []
    best_data = []
    worst_data = []
    average_data = []
    median_data = []
    std_data = []
    dimension = dim[h]
    print('Dimension: ', dimension)
    for i in range(len(optimizer_list)):
        optimizer = optimizer_list[i]
        for j in range(len(function_list)):
            function, f_id = func(function_list[j], dimension)
            
            for k in range(num_run):
                x = opt(optimizer, population_size, function)
                x.initial_solutions()
                
                C_iter = 1
                while C_iter < M_iter:
                    start_time = time.time() 
                    x.update_position(M_iter)
                    x.sort_population()
                    total_time = time.time() - start_time
                
                    optimizer_data.append(optimizer)
                    dimension_data.append(dimension)
                    function_data.append(function_list[j])
                    function_id_data.append(f_id)
                    run_data.append(k+1)
                    time_data.append(total_time)
                    iteration_data.append(C_iter)
                    best_data.append(x.get_global_best())
                    worst_data.append(x.get_global_worst())
                    average_data.append(x.average_result())
                    median_data.append(x.median_result())
                    std_data.append(x.std_result())
                    
                    C_iter = C_iter + 1                
                
# create data frame for the csv file
    df = pd.DataFrame({'Optimizer': optimizer_data, 
                       'Dimension': dimension_data,
                       'Function': function_data,
                       'Function ID': function_id_data,
                       'Run': run_data,
                       'Time': time_data,
                       'Iteration': iteration_data,
                       'Best': best_data,
                       'Worst': worst_data,
                       'Average': average_data,
                       'Median': median_data,
                       'Std': std_data
                       })
    
    del optimizer_data, dimension_data, function_data, function_id_data, run_data, time_data, iteration_data, best_data, worst_data, average_data, median_data, std_data
    df.to_csv('experimentAOA{}.csv'.format(dimension))

print ('Done')
