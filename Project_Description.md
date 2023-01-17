# SVR and Simulated Annealing

Support Vector Regression (SVR);
+ is a type of algorithm that can be used for both __regression and classification problems__.
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

# Annealing



