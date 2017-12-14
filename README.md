# Spark Playground
This repo holds a couple of my personal Spark projects.

# Models
### Logistic Regression
Currently implemented with some random features from the data set, but the initial evaluation numbers have already improved with more correct feature engineering (one hot encoding for categorical data). `Creditability` is the class we're trying to predict. Current numbers:
```
Accuracy = 0.7284768211920529
Confusion matrix:
27.0  51.0
31.0  193.0
Precision(0.0) = 0.46551724137931033
Precision(1.0) = 0.7909836065573771
Recall(0.0) = 0.34615384615384615
Recall(1.0) = 0.8616071428571429
```
Not great, but probably fair for one day, and will hopefully improve. Obviously, more explanation is needed in this section, as well.

# Credits
##### German Credit Data
* `data/german_credit.csv`: Penn State [link](https://onlinecourses.science.psu.edu/stat857/sites/onlinecourses.science.psu.edu.stat857/files/german_credit.csv)

I believe this is an adaptation of the UCI set, originally provided by Dr. Hans Hofmann of the Universitat at Hamburg, in 2000, found [here](https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)).
