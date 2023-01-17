# Forecasting MTCOP Prices with Support Vector Regression (SVR) through Simulated Annealing (will be updated on its context)
This project showcased on how a metaheuristic algorithm, Simulated Annealing is implemented in Support Vector Regression (SVR) for hyperparameter optimization to forecast the next Malaysian Tapis Crude Oil (MTCO) Price.


https://github.com/ArdaniahJ/Forecasting_MTCO_Prices_with_SVR_through_Simulated_Annealing.wiki.git

## So..why SVR?
SVR-based models __do not suffer from over-fitting problems__. ie; _high accuracy for training dataset and low for test dataset;_ as they are based on structural risk minimization (SRM) principle `(tuning the capacity of the classifier to the available amount of training data)` which always guarantee a unique, global and optimal solution.

### Why Optimization is Used in SVR?
SVM can gives good prediction even with fewer amount of data, the only disadvantage of using SVR technique is that it `requires an efficient parameter optimization to give prediction with higher accuracy.`

SVR model parameters nee to be optimized via some optimization technique such as:
1. Grid search methodology with 10-fold cross-validation
2. Particle swarm optimization (PSO)
3. Ggenetic algorithm (GA)
4. Differential evolution (DE)
5. Simulated Annealing (SA) & etc

The more robust the optimization the SVR model parameters, the better is the prediction accuracy and generalization of the SVR model. 

## Annealing
It comes from the technique of `Annealing` in metallurgy. 
  + It's a technique where metal is heated to high temperature and slowly cooled down to improve its pysical properties. 
  + When the metal is hot, the molecules randomly rearrange themselves at a rapid pace. If it being cooled down abruptly, there will be more bubbles and the molecules moves are instabilized. 
  + This is why the metal is being cooled slowly as the rearrangment of the molecules occurs in accordance to the speed of the cooling process until it becomes stabilized, hence the resultant metal will be desired workable metal. 
  + The factors of time and metal's energy at a particular time will supervise the entire process. 
  
### Simulated Annealing Algorithms
In ML, Simulated Annealing Algorithm mimics the Annealing process and is used to find optimal (or most predictive) features in the feature selection process. 

Heuristic and metaheuristic optimization techniques have played a significant role in managing and providing better performance solutions. 
  + `Heuristic` : is a technique aimed to solve a problem faster when traditional techniques are too slow
  + `Metaheuristic` : is a higher-level techninque or heuristic that seeks, generates, or selects a heuristic that may provide a sufficiently good solution to an optimization problem. 
    + it's also a way to find solutions to optimization problem. In constrast to other optimization procedures, they are often motivated more by analogy to some process in nature (in this case 'Annealing') and don't guarantee finding globally optimal solutions. 
    + On the other hand, they work in much broader contexts compared to other optimization algorithms like linear programming. 
    + The constrains on the cost function are much more lenient (a convex cost function is not needed for stuff like simulated annealing to work).
    + As such, metaheuristic can be and are (mostly stochastic gradient descent of Neural Network [NN]) used to train machine learning models. But, metaheuristics are also used for other optimization problems outside of ML.From `energy-minimization-based-simulations in physics` to `combinatorial-optimization-problems in operations research`.
 
 When to consider Simulated Annealing?
 It is an ideal candidate if several parameters of the same type needs to be optimized at once (vectors). I personally use this algorithm when __I want to reach at least a satisfactory solution extremely quickly.__ If I want the solution to be as accurate as possible, I turn to genetic algorithm.
  
 # Algorithm
 <p align="center"><img src = "water sort ring game.jpeg" width = "250"  /></p>

 
As with other optimization algorithms, we will create a random solution.
However, its optimization will consist in lowering the temperature a pre-defined number of times (n) and based on this temperature and the generated neighbour `we can accept or reject it. `. A simplified form of the algorithm would work like this;
 1. `solution` = __random solution__
 2. `temperature` = __initial temperature__
 3. `n times`:
    + `neighbour_solution` = __neighbourhood solution based on solutions__
    + `sneighbouring_solution` = __is better than__ `solution`
      + true --> `solution` = __neighbour_solution__
      + false --> `solution` = __neighbor_solution__ if temperature is high enough
 4. __lower temperature__
 5. __Return best__ ` solution` __found yet__
 
 ### When is the temperature is high enough to accept the `neighbour_solution`? 
 This is where the classic formula that the observation of annealing brough us comes into play. We accept `neighbour_solution_ with probability:
 ```
 P(∆E)=e^{\frac{-∆E}{t}}
 ```
 
 Where:
 * ∆E = score difference of the neighbouring solution and the original solution. 
 
 Example of implementation of the formula;
 ```
 fun accept(actual: Solution, neighbor: Solution, temp: Double) : Boolean {
    val diff = neighbor.cost - actual.cost
    if (diff < 0) { return true }
    return random.nextDouble() < exp(-diff / temp)
 ```
 
 ### How to lower the temperature?
 We can reduce the temperature in different ways such as:
 1. Linear temperature reduction: `T' = T - α`
 2. Geometric temperature reduction: `T' = T . α`
 3. Slow temperature reduction: `T' = T/(1+βT)`
 

### Variants 
There are 2 main cases of using this algo:
1. We get a neighbour with a small change and accept him with a certain probability
2. The higher the temperature, the more significant changes we can make in the neighbour. 

Furthermore, the algorithm can vary, for example, in when we change the temperature. Someone changes the temperature in each iteration, someone lets the algorithm to cool down at every n-iterations. 

No one can simply say how to set the initial temperature, and no one can say how fast to let the temperature to cool down (value of α).

#

# Why use SVR to forecast market price?
Support Vector Regression (SVR) is a type of algorithm that can be used for both regression and classification problems. It is a type of support vector machine that is often used for forecasting market prices because it has a few characteristics that make it well-suited to this task.

SVR can handle non-linear data: The stock market is a complex system with many factors that can influence prices, and these factors may not always have a linear relationship. SVR is a non-linear algorithm, which means it can handle complex relationships between inputs and outputs.

SVR can handle high dimensional data: The stock market has a lot of variables such as economic indicators, company-specific information, and news events that can influence prices. SVR can handle high dimensional data, which means it can take into account many variables when making predictions.

SVR can handle large datasets: The stock market generates a large amount of data, and SVR can handle large datasets.

SVR can handle outliers: Stock prices are often affected by unpredictable events, such as natural disasters or political crises, that can cause extreme fluctuations. SVR is robust to outliers, which means it is less affected by these extreme fluctuations than other algorithms.

SVR can handle a large margin of error: In stock price prediction, the margin of error between prediction and actual value can be huge, SVR can be used to handle large margin of error.

All of these characteristics make SVR a good choice for forecasting stock prices. It is able to handle the non-linear, high-dimensional, and large-scale nature of stock market data, as well as being robust to outliers and large margin of error.

# Code Explanations
This code is a machine learning program that uses a technique called Support Vector Regression (SVR) to make predictions. SVR is a type of algorithm that can be used for both regression and classification problems. The program starts by importing several libraries, such as pandas, numpy, and sklearn, which are libraries commonly used for machine learning. The program then defines some functions that will be used later on in the code.

The first function, RegressionMetric, takes in two inputs: valid data and predictions. It calculates four different metrics (MAE, MAPE, MSE, and R2) to evaluate the performance of the predictions. These metrics are used to measure the difference between the actual values and the predicted values.

The second function, model_training, is used to train the model and output the best model with the specified metric. This function takes in several inputs, such as Xtrain, Ytrain, Xvalid, Yvalid, current_parameters, and constant_parameters. The current_parameters and constant_parameters are hyperparameters that are used to control the behavior of the model. The function creates an instance of SVR with the specified hyperparameters, trains the model using the Xtrain and Ytrain data, and makes predictions using the Xvalid data. The function then calculates the metric_value using the MSE function to measure the performance of the predictions.

The third function, parameters_choosing, is used to choose parameters for the next iteration. It takes in two inputs: potential_parameters, and current_parameters. The function randomly chooses a parameter from potential_parameters and updates it with a new value. The function then returns the updated parameters.

The code then uses these functions in a loop that runs several times to find the best combination of hyperparameters that results in the best predictions. It repeatedly updates the hyperparameters, trains the model, and makes predictions using the new hyperparameters, until a satisfactory level of accuracy is achieved.

