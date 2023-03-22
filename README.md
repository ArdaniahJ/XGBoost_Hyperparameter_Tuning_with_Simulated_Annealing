<h1 align="center"> Revolutionizing Fraud Detection: <br> A Simulated Annealing Algorithm for Optimizing XGBoost Hyperparameters </h1>

<p align="center">Banking Subset Fradulent Case: Credit Card Holder </p> 

## Overview  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1kWKL8elUpdElyh6YcCdXCKAPKrgdqgdX?usp=sharing)

This repository contains the code and implementation details for using simulated annealing to tune hyperparameters for XGBoost in the context of detecting financial fraud credit card transactions. This approach can help improve the performance of XGBoost and enhance the model's ability to detect fraudulent transactions.

### Expected Industry Tangible Outcome
Optimized financial fraud detection model by tuning XGBoost hyperparameters with Simulated Annealing with a target to achieve 10% accuracy improvements and 5% reduction in fraud losses for a banking client, and over $1M in cost savings returns over the course of a year.

## Dataset
The dataset used in this project is the Credit Card Fraud Detection from: [ceicdata](https://www.ceicdata.com/en/malaysia/credit-card-statistics). 

It contains the transactions:
+ made by credit cards in Sept 2013 by European cardholders.
+ occurence within two days - 492 frauds out of 284807 transactions.

The data is highly imbalanced, with the positive class (frauds) accounting only for 0.172% of all transaction. 

## Requirement 
In Colab, install the updated version of the packages;
```python
!pip install --upgrade xgboost numpy scipy
```

## Code Explanation
+ In `SA.py` there are few functions;
  + `model_training` : to train the model
  + `choose_params`: to choose set of parameters from the vicinity of the current parameters. 
  + `simulate_annealing`: to optimize the hyperparameters chosen from `choose_pparams` using SA technique.
+ To run in Colab, run the notebook overall as it has been amended to tend to the colab env.
  + 👉🏻 p/s: In case of perpetual training, run the codes in `python file` only. 
  ```bash
  # in main.py
  from utils.SA import simulate_annealing * 
  ```

The hyperparaemeters of XGBoost tuned are:
  1. `max_depth` - specifies the max depth to which each tree will be built
  2. `subsample` - fraction of observations used to train individual learners
  3. `colsample_bytree` - fraction of columns considered for each split
  4. `learning_rate` - shrinkage weights of weights
  5. `gamma` - min loss reduction required to make a split (split is made when gamma is reached)
  6. `scale_pos_weight` - controls balance of +ve and -ve ration
  
The parameters search space for the tuned hyperparameters is as below, it can be amended according to computational limitation & data complexity:
```python
param_dict = OrderedDict()
param_dict['max_depth'] = [5, 10, 15, 20, 25]
param_dict['subsample'] = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
param_dict['colsample_bytree'] = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
param_dict['learning_rate'] = [0.01, 0.05, 0.10, 0.20, 0.30, 0.40]
param_dict['gamma'] = [0.00, 0.05, 0.10, 0.15, 0.20]
param_dict['scale_pos_weight'] = [30, 40, 50, 300, 400, 500, 600, 700]
```
## Run the Simulated Annealing
```python
final_res, bestest_model = simulate_annealing(param_dict, constant_params, xtrain, xvalid, ytrain, yvalid, model_training) 
```

## Update 
The `SA_updated.py` is updated with `metropolis_formula` & `total_run_calculator` to organize the code to reduce the computational memory. One can add below functions in `SA.py` or use `SA_updated.py` ultimately. 

```python
def metropolis_formula(temperature, current_metric, previous_metric):
    random_number = np.random.uniform()
    difference = current_metric - previous_metric
    metropolis_val = np.exp(- difference / temperature)
    return random_number, difference, metropolis_val

def total_run_calculator(minimum_temperature, initial_temperature, alpha, iterations_number):
    total_no_temp_drop = np.log(minimum_temperature/initial_temperature)/np.log(alpha)
    total_iterations_number = ceil(total_no_temp_drop)*iterations_number
    return total_iterations_number 
```

### Future recommendation
- [ ] __Visualize Optimization__: Make a moving animation route on how the hyperparameters are tuned. Eg; picture below (taken from SA case study: TSP)

<p align="center">
  <img src="https://camo.githubusercontent.com/d8bf68b17ce7e7d3303e468ea0b98650d9e5edb98277678bc4df9d3cab5d67b2/68747470733a2f2f6d656469612e67697068792e636f6d2f6d656469612f336f686a554f4e6679354971626158316b592f67697068792e676966" class="centerImage" alt="EG SA" height="300" width="350" />
</p>

Copyright © 2023, [Ardaniah Jamaluddin](https://github.com/ArdaniahJ). Released under the [GNUv3.0](https://github.com/ArdaniahJ/Forecasting_MTCO_Prices_with_SVR_through_Simulated_Annealing/blob/c67a9922748cfda3f987c824683b36306d358009/LICENSE)
