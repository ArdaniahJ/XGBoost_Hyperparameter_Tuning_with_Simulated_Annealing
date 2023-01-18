# SVR and Simulated Annealing

Support Vector Regression (SVR);
+ Is a type of algorithm that can be used for both __regression and classification problems__.
+ SVM is often used in time series forecasting due to its characteristics:
    + SVR can handle __non-linear data__: Ie; Stock Market is a complex system with many factors that can influcence prices, and these factors may not always have a linear relationship. SVR is a non-linear algorithm, which means it can handle complex relationship between inputs and outputs. 
    + SVR can handle __high-dimensional data__: ie; The stock market has lots of variables that influences the prices such as;
        + economic indicators
        + company-specific information
        + news-event 
        + natural disasters
        + political crises
<br> It takes all the many high-dimensional data with these variable into account when making predictions <br>
    + SVR can handle __large datasets__
    + SVR can handle __outliers__: ie; Stock prices are often affected by unpredictable events that can cause _extreme fluctuations_. SVR is robust to outliers, thus being less affected by these extreme fluctuations than other algorithm. 
    + SVR can handle __a large margin of error__: handling a huge margin of error between prediction and actual value
    
## So...why SVR for this project?
SVR-based models __do not suffer from overfitting problems__ (_high accuracy for training dataset and low accuracy for test dataset_) as they are based on __Structual Risk Minimization (SRM) principle__[^1].  (_tuning the capacity of the classifier to the available amount of training data_) which always `guarantee a unique, global and optimal solution.`


[^1]: SRM principle addresses overfitting problem by balancing the model's complexity against its success at fitting the training data. This gives better results than empricial risk minimization (ERM).

### Annealing



### Why is hyperparameter optimization needed in SVR?
Hyperparameter optimization is necessary in SVR because the performance of the model can vary significantly depending on the choice of the hyperparameters.

Hyperparameters are parameters that are not learned during training, but are set before training begins. In this case of SVR, common hyperparameters include the;
+ __kernel type__: determines the type of function used to map the input data into a higher dimensional space where it becomes linearly separable. Common kernel types include;
    + linear
    + polynomial
    + radial basis function (RBF)
+ __the regularization parameter (C)__: controls the trade-off between maximizing the margin and minimizing the misclassification error. 
    + `SVM : min |w|^2+C∑ξ`
    + _if C is too large_, the optimization algo will try to reduce `|w|` as much as possible. this leads to hyperplane which tries to classify each training example correctly resulting to the loss in generalization properties of the classifier. `tldr; overfit will happen here`
    + _if C is too small_, the objective function will have its certain freedom to increase |w| a lot thus leading to a large margin error. `tldf; the classifier will underfit. it doesn't segregate the datapoints into their respective classes`
+ __gamma parameter__: controls the width of the RBF kernel. 
    + _the &darr; (smaller) the gamma value &uarr; (wider) the RBF kernel._ `overfitting will happen`
    + _the &uarr; (larger) the gamma value; the &darr; (narrower) the RBF kernel._ `underfitting will happen`

Hence, finding the optimal combination of hyperparameters is crucial for the SVR model to perform well on the data.  By using SA technique, it's possible to find the best set of hyperparameters that;
    + minimize the evaluation metrics (less losses)
    + improve the model performance
    
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







