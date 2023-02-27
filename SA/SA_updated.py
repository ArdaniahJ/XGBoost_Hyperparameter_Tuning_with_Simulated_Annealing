from collections import OrderedDict # to preserve the order in which key-value pairs (items are traversed in the original order)
from random import random
from math import ceil
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

import matplotlib.pyplot as plt
%matplotlib inline

# function to choose sets of parameters for the next iteration
def choose_params(potential_parameters, current_parameters = None):

  # if current parameters is non-empty dict
  if current_parameters:
    #copy current parameters dict to another dict called "new parameters"
    new_parameters = current_parameters.copy()
    #change the "keys" of the potential parameters dict into list
    potential_param_keys = [*potential_parameters.keys()]
    #randomly choose a "key" out of the potential_param_keys and set the "key" as the parameter-to-be-updated
    param_to_update = np.random.choice(potential_param_keys)
    #obtain the potential values for the parameter-to-be-updated
    potential_param_values = potential_parameters[param_to_update]
    #find the current index of the value of the parameter-to-be-updated
    curr_param_val_index = potential_param_values.index(current_parameters[param_to_update])
    no_potential_param_val = len(potential_param_values)
    random_range = tuple(x for x in np.arange(-no_potential_param_val, no_potential_param_val + 1) if x != 0)
    
    #if the value of the parameter is the first in the list of potential values 
    if curr_param_val_index == 0:
      positive_random_index = np.random.choice([x for x in random_range if x > 0])
      #set the value of the parameter-to-be-updated as the second in the list of potential values
      new_parameters[param_to_update] = potential_param_values[positive_random_index - 1]
    #if the value of the parameter is the last in the list of potential values 
    elif curr_param_val_index == len(potential_param_values) - 1:
      negative_random_index = np.random.choice([x for x in random_range if x < 0])
      #set the value of the parameter-to-be-updated as the second last in the list of potential values
      new_parameters[param_to_update] = potential_param_values[negative_random_index + 1]
    else:
      #set the value of the parameter-to-be-updated as the value with index +1 or -1 in the list of potential values
      restrict = np.arange(ceil(len(random_range)*0.25))
      final_index = curr_param_val_index + np.random.choice(restrict)
      
      if final_index >= no_potential_param_val: 
        new_parameters[param_to_update] = potential_param_values[-1]
      elif final_index <= 0:
        new_parameters[param_to_update] = potential_param_values[0]
      else:
        new_parameters[param_to_update] = potential_param_values[final_index]
   
  # if the current parameters is empty dict
  else:
    #create a new empty dict
    new_parameters = {}
    #randomly assign the potential values to the parameters
    for k, v in potential_parameters.items():
      new_parameters[k] = np.random.choice(v)

  return new_parameters


def metropolis_formula(temperature, current_metric, previous_metric):
    random_number = np.random.uniform()
    difference = current_metric - previous_metric
    metropolis_val = np.exp(- difference / temperature)
    return random_number, difference, metropolis_val


def total_run_calculator(minimum_temperature, initial_temperature, alpha, iterations_number):
    total_no_temp_drop = np.log(minimum_temperature/initial_temperature)/np.log(alpha)
    total_iterations_number = ceil(total_no_temp_drop)*iterations_number
    return total_iterations_number


def simulate_annealing(param_dict, constant_params, 
                        X_train, Y_train,  X_valid, Y_valid, 
                        no_iters: int = 8, alpha = 0.95,  
                        initial_temperature = 50, min_temperature = 10):

    T = initial_temperature
    T_min = min_temperature

    columns_name = ['Number of Temperature Reduction'] + [*param_dict.keys()] + ['Metric', 'Best Metric']
    results = pd.DataFrame(columns = columns_name)
    _, ori_metric = model_training(X_train, Y_train, X_valid, Y_valid)

    prev_params = None
    prev_metric = ori_metric
    best_metric = ori_metric
    hash_values = set()
    result_list = []
    j = 0

    while T >= T_min:
        print("Current Temperature is: %.2f" %T)
        print("\n")
        
        for i in range(no_iters):
            print('Starting Iteration ' + str(i + 1))

            curr_params = choose_params(param_dict, prev_params)
            hash_val = tuple(curr_params.values())

            if hash_val in hash_values:
                print('Combination revisited.')
                print('\n\n')

            else:
                hash_values.add(hash_val)
                            
                model, metric = model_training(X_train, Y_train, X_valid, Y_valid, 
                                            curr_params, constant_params)

                if metric < prev_metric:
                    print('Local Improvement in metric from {:8.6f} to {:8.6f} '
                            .format(prev_metric, metric) + ' - parameters accepted' + '\n')
                    prev_metric = metric
                    prev_params = curr_params.copy()
                    
                    if metric < best_metric:
                        print('Global Improvement in metric from {:8.6f} to {:8.6f} '
                                .format(best_metric, metric) + ' - best parameters updated' + '\n\n')
                        best_metric = metric
                        best_model = model
                
                else:
                    random_no, diff, Metropolis= metropolis_formula(T,metric, prev_metric)
                    
                    if random_no < Metropolis:
                        print("No Improvement but parameters are ACCEPTED.") 
                        prev_metric = metric
                        prev_params = curr_params
                        
                    else:
                        print("No Improvement and parameters are REJECTED.") 
                    
                    print("Metric change:   %.6f" % diff)
                    print("Threshold:       %.6f" % Metropolis)
                    print("Random Number:   %.6f" % random_no)
                    print('\n')

            results.loc[i, 'Number of Temperature Reduction'] = j
            results.loc[i, [*curr_params.keys()]] = [*curr_params.values()]
            results.loc[i, 'Metric'] = metric
            results.loc[i, 'Best Metric'] = best_metric
            print("\n")
        
        result_copy = results.copy()
        result_list.append(result_copy)
        
        T = alpha * T

        print("Temperature has been reduced.")
        print("The number of temperature reduction: " + str(j + 1))
        j = j + 1

        if T < T_min: print("Minimum temperature is reached. The algorithm is terminated.")

        final_result = pd.concat(result_list)

        total_no_iter = total_run_calculator(min_temperature, initial_temperature, alpha, no_iters)
        
    return best_model, final_result, total_no_iter