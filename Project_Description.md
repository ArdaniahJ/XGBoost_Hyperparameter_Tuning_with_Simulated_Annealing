# XGBoost and Simulated Annealing

[^1]: SRM principle addresses overfitting problem by balancing the model's complexity against its success at fitting the training data. This gives better results than empricial risk minimization (ERM).
[^2]:  In SA, the temperature variable is used to simulate the annealing heating process. 

### Annealing & Simulated Annealing
+ __Annealing__: comes from metallurgy. involves heating and cooling a metal/material to alter its physical properties due to the changes in its internal structure. as the metal cools, its new structure becomes fixed, consequently causing the mtal to retain its newly obtained properties.

<p align="center">
  <img src="https://github.com/ArdaniahJ/Forecasting_MTCO_Prices_with_SVR_through_Simulated_Annealing/blob/afc231b20e47ec56a34816c69f18c2cc9ed2b6a2/water%20sort%20ring%20game.jpeg" width=250 />
</p>

+ __Simulated Annealing__: is _a way of finding the best solution to a problem_, just like how a metalworker finds the best shape for a piece of metal by heating it up and then cooling it down[^2]
    + Take the water bubble rings toy as an analogy. when then buttons are pressed, the rings move inside the water. the aim of the game is _to get as many rings hooked possible_. 
    + But, how to play? intuitively, one would first start by pressing the buttons vigorously and by eyeballing to see if all the rings hooked without worrying about the proper solution; just randomly play.
    +  once some of the rings hooked in the hooks, one tend to be more careful. after that, a serious and deliberate move will follow suits. eventually when the final steps to win appears, one will be extremely cautious and only move after a careful and gracious plan. 
    +  __That's the intuition behind the SA__.
    +  At the beginning, you dont care if youre actually moving towards the good solution and you accept bad moves with bad configurations as well. 
    +  But as you progress towards the solution, you tend to be more careful and only selecting good moves.
    +  to formulate any problem in order to apply algo like SA, every problems needs to have a way to define how good a given configuration is. 
    +  for this particular toy game, given the configuration of this toy, one can tell how good the solution is by checking the number of rings that are hooked. 
    +  hence, every system or every problem that to be solved needs to have a way to describe the goodness or the fitness of the solution for the problem.
    +  in algo words, we need to have the mechanism to define the energy of the system.
    +  taking this formula into consideration <br> `P(ΔE) = e^{-(ΔE/k.t)}` or `P(E(next) - E(current)) = e^(-deltaE/T)` where;
        + __E__: _energy of the system_. 
            + everytime the system mechanism goes through a change of config, the change in energy [delta E (ΔE)] can be computed.
    + in the toy game, one can play the game in a specific way (ie; focus only for one hook). if that method is used, one will end up in a deadlock where only one hook will have lots of rings but the other is empty. 
    + however, one may play vigorously but this'll removes lots of hooked rings leading to decreased in playing tempo but takes a longer time to finish instead.  this describes `local maximum or local minimum`.
    + when all possbile configs of a given system are plotted on X & y axes, the corresponding energy are able to be computed. Energy landscape will look like this (refer the formula).
        + _&uarr; (high) values correspond to &uarr; (maximum) energy_. in the hooked rings case, since max hooked rings is desired; hence this is a __maximization problem__ and max value of energy is needed for this particular problem. 
        + places like these in the E landscape where the energy is &uarr; (_high_) are known as `local maxima or local minima`.
        + these are the points where the system can get stuck and cannot find the global maximum.
        + so, SA algo is designed to avoid getting stuck in the local maxima or minima by __introducing a probability of accepting bad moves or bad configs__. this is known as the `acceptance probability`. 
        + as the algorithm progresses, the acceptance probability &darr; (_decreases_), and becomes more careful and deliberate in its moves.





    + the temperature, __θ__ is initially set to &uarr; (_high_). It then allows to slowly to &darr; (_cool_) as the algorithm runs. 
    + when the __θ__ is &uarr; (_high_), the algo'll be allowed with more freq to accept solutions that are worse than the current solution. _this gives the ability to jump out of any local optimus it finds itself in early execution_
    + as the __θ__ &darr; (_reduced_), so is the chane of accepting worse solutions, therefpre allowing the algo to gradually focus on area of search space in which hopefully, a close to optimum solution can be found.
    + this gradual cooling process is what makes the SA algo remarkbly effective at finding a close to optimum solution when dealing with large problems which contain numerious local optimums. 
    
### Metrics used in this project
Since this is a time series regression problem, thus a metric for regression as below is used to evaluate the performance of the model;
1. __MAE__: measures the average absolute difference between pred and actual values.
    + _&darr; (closer) to 0, the &uarr; (higher) accuracy of the model_ <br>`MAE = ( ∑|actual - pred|)/#datapoints`
2. __MSE__: measures the average squared difference between pred and actual values. 
    + _&darr; (closer) to 0, the &uarr; (higher) accuracy of the model_ <br>`MSE =  ∑(actual- pred)^2/#datapoints`
3. __R2__: measures of how well the model fits the data, with a value of 1 inidicating a perfect fit. 
    + it doesn't measure the accuracy of the predictions but the correlation of X and y of the model.
    + _the &uarr; (higher) the R2, the &uarr; (better) the model_
    + if R2 is 1, it means all variation of y is explained by X (tho it's rare in real world)
    + __R2 > 0.75 is considered a strong correlation__ <br>`R2 = 1 - ( ∑(y_pred - y_mean)^2/∑(y_actual - y_mean)^2)`
4. __RMSE__: is the square root of the MSE. 
    + Squared error (__L2__) is a row-level error calc where the _(pred - actual)_ is squared. 
    + RMSE helps to understand the model performance over the whole dataset. 
    + it's in the same scale as the pred unit. 
    + it ranges _from 0 to infinity_, _&darr; (closer) to 0, the &uarr; (higher) the accuracy of the model_ <br>`RMSE = √ MSE`
5. __MAPE__: is the ∑ of MAE. 
    + used when communicating with end users
    + not advisable to use when _actual values are closer to 0 due to the division by zero error_
    + general rule of thumb for MAPE is as below; <br>`MAPE = {1/#datapoints . (∑|actual - pred|/actual)} x 100%`

| MAPE  | Interpretation|
|---|---|
| < 10%   | Very good  |
| 10 - 20%  | Good |
| 20 - 50%  | OK  |
| > 50%  | Not good |


#
SA finds the __subset of features__ that gives the __best predictive model performance__ amongst many possible subset combinations.

SA improves SVR strategy through introduction of two elements;
1. Metropolis algorithm - 
    + in which some states that do not improve energy are accepted when they serve to allow the solver to explore more of the possible space of solutions. 
    + such "bad" states are allowed using the Boltzman criterion ` e−ΔJ/T > rnd(0, 1)` where;
        + ΔJ is the change of energy
        + T is a temperature 
        + rnd(0,1) is a random number in the interval [0,1)
        + J = cost function. it corresponds to the free energy in the case of annealing.
        + if T is large, many "bad" states are accepted and a large part of solution spaces is accessed. 

2. Lower the Temperature.
    + After visiting many states and observing that the J (cost function) declies only slowly;
        + one lowers the temperature; thus limits the size of allowed "bad" states.
    + After lowering the temperature several times to a low value;
        + one may then "quench" the process by accepting only "good" states in order to find the local minimum of the J.

# 
The elements of SA are;
+ A finite set of S
+ A cost function, J, defined on S. Let S* ⊂ S be the global minima of J
+ For each i ∈ S, a set S(i) ⊂ S − i is called the set of neighbors of i.
+ For every i, a collection of positive coefficients qij, j ∈ S(i), such that Σj∈S(i) qij = 1.
    + It is assumed that j ∈ S(i) if and only if i ∈ S(j).
+ A nonincreasing function T : N → (0,∞), called the cooling schedule. Here N is the set of positive integers, and T(t) is called the temperature al time t.
+ An initial state x(0) ∈ S.

The SA algo consists of a discrete time inhomogeneus Markov Chain x(t).
+ If the current x(t) = i; choose a neighbor j of i at random; 
    + the probability that any particular  j ∈ S(i) is selected is equal to qij.
    + once j is chosen, then next state x(t+1) is determined as follows;
    ```
    if J(j)≤J(i) then x(t+1)=j
    if J(j)≤J(i) then x(t+1)=j with probability e−(J(j)−J(i))/T(t)%
    else x(t+1)=i
    ```
    
    Metropolis cycle is repeated until thermal equilibrium is reached. 
    
# Code Explanation

In this project, a `non-greedy` approach driven by the randomness imparted in the algorithm will be used. 
+ this approach can reassess previous subsets and move in a direction that is initially unfavourable if there appears to be a potential benefit in the long run. 
+ as a result, simulated annealing has a greater ability to escape local optima and find the global best subset of features. 
+ it also creates a higher probability of capturing key features interaction, especially if these interactions are significant only if the features co-occur in a subset. 

1. Generate a random initial subset of features to form the current state. This step can be done arbitrarily by randomly selecting 50% of the original features. 
2. Evaluate model performance on the current subset to get an intial performance metric.
`def model_training` runs a SVR on chosen current & constant parameters to obtain the F1 Score; where lower scores is better, hence `metric < prev_metric` (current metric must be lower than the prev metric).

[^3]: https://towardsdatascience.com/feature-selection-with-simulated-annealing-in-python-clearly-explained-1808db14f8fa
