# SVR and Simulated Annealing

## So...why SVR?
SVR-based models __do not suffer from overfitting problems__ (_high accuracy for training dataset and low accuracy for test dataset_) as they are based on __Structual Risk Minimization (SRM) principle__[^1].  (_tuning the capacity of the classifier to the available amount of training data_) which always `guarantee a unique, global and optimal solution.`


[^1]: SRM principle addresses overfitting problem by balancing the model's complexity against its success at fitting the training data. This gives better results than empricial risk minimization (ERM).

