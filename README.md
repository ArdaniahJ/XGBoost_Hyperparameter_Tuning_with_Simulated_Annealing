# Forecasting MTCOP Prices with Support Vector Regression (SVR) through Simulated Annealing (will be updated on its context)
Hyperparameter optimization of Support Vector Regression (SVR) using a metaheuristic algorithm, Simulated Annealing (SA) applied in Malaysia's Tapis Crude Oil Price (MTCOP) prediction

# Why Optimization is Used in SVM (Support Vector Regression)?
ML involves using an algorithm to learn and generalized from historical data in order to make predictions on new data. 
<br>
This problem can be described as `approximating a function that maps examples of inputs to examples of outputs`. Approximating a function can be solved by framing the problem as `function optimization`. 
<br>
This is where ML algo defines a;
  + `parameterized mapping function` (eg; a weighted sum of inputs)
  + and an `optimization algorithm` ;
  <n>
to fund the values of the parameters (eg; model coefficients) that minimize the error of the function when used the map inputs to outputs. 

This means that each time the model is `.fit` for training, an optimization problem is solved. 



# Annealing
It comes from the technique of `Annealing` in metallurgy. 
  + It's a technique where metal is heated to high temperature and slowly cooled down to improve its pysical properties. 
  + When the metal is hot, the molecules randomly rearrange themselves at a rapid pace. If it being cooled down abruptly, there will be more bubbles and the molecules moves are instabilized. 
  + This is why the metal is being cooled slowly as the rearrangment of the molecules occurs in accordance to the speed of the cooling process until it becomes stabilized, hence the resultant metal will be desired workable metal. 
  + The factors of time and metal's energy at a particular time will supervise the entire process. 
  
# Metaheuristic: Simulated Annealing Algorithms
In ML, Simulated Annealing Algorithm mimics the Annealing process and is used to find optimal (or most predictive) features in the feature selection process. 

Heuristic and metaheuristic optimization techniques have played a significant role in managing and providing better performance solutions. 
  + `Heuristic` : is a technique aimed to solve a problem faster when traditional techniques are too slow
  + `Metaheuristic` : is a higher-level techninque or heuristic that seeks, generates, or selects a heuristic that may provide a sufficiently good solution to an optimization problem. 
    + it's also a way to find solutions to optimization problem. In constrast to other optimization procedures, they are often motivated more by analogy to some process in nature (in this case 'Annealing') and don't guarantee finding globally optimal solutions. 
    + On the other hand, they work in much broader contexts compared to other optimization algorithms like linear programming. 
    + The constrains on the cost function are much more lenient (a convex cost function is not needed for stuff like simulated annealing to work).
    + As such, metaheuristic can be and are (mostly stochastic gradient descent of Neural Network [NN]) used to train machine learning models. But, metaheuristics are also used for other optimization problems outside of ML.From `energy-minimization-based-simulations in physics` to `combinatorial-optimization-problems in operations research`.
    
