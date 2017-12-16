# Spark Playground
This repo holds a couple of my personal Spark projects.

# Installation and Packaging
1. Install Apach Spark 2.2.x (compiled with 2.2.0):
  * On a Mac: `brew install spark`
  * Else, download, untar and add the `spark/bin` directory to `$PATH`
2. Checkout this repository:
  * `git clone https://github.com/mmatthews06/spark-playground.git`
3. If using IntelliJ:
  * Open this project. The defaults probably work just fine.
  * Allow IntelliJ to download relevant libraries listed in `build.sbt`. This should startup automatically.
  * Add a run configuration for `sbt package`
    * Open **Run -> Edit Configurations**
    * Click the **+** to add a new configuration.
    * Pick **sbt Task**
    * Set *Tasks* to `package`
    * Name it anything you want (e.g., `sbt package`)
    * Hit *Ok* with the defaults (only 2 fields changed)
  * Click *Run* to build a `.jar` file, `spark-playground/target/scala-2.11/spark-playground_2.11-0.1.jar`
4. If not using IntelliJ, installing `sbt` with `brew` should work. Then run `sbt package` in the spark-playground directory.

# Execution
Running the current tasks should be as simple submitting the `.jar` file using `spark-submit`, with the `data/german_credit.csv` file as the first argument. Example steps:
1. In a shell, navigate to the directory with the `.jar` file:
```bash
cd spark-playground/target/scala-2.11
```
2. Submit the job. Note, `local[K]` specifies the number of worker threads to use, where `*` selects as many as possible:
```bash
spark-submit --master local[*] spark-playground_2.11-0.1.jar ../../data/german_credit.csv
```

# Models
### Classification
##### Logistic Regression
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

##### Decision Tree
As with the others, this is a first implementation. The very first evaluation numbers are worse than logistic regression. Again, these are randomly picked features.
```
---------- Decision Tree ----------
Accuracy = 0.7086092715231788
Confusion matrix:
34.0  44.0
44.0  180.0
Precision(0.0) = 0.4358974358974359
Precision(1.0) = 0.8035714285714286
Recall(0.0) = 0.4358974358974359
Recall(1.0) = 0.8035714285714286
```

### Misc.
##### Principal Component Analysis
For future feature engineering, this is an implementation of PCA. As with the others, this is a "first implementation" stage. It executes, and prints some information out.

# Credits
##### German Credit Data
* `data/german_credit.csv`: Penn State [link](https://onlinecourses.science.psu.edu/stat857/sites/onlinecourses.science.psu.edu.stat857/files/german_credit.csv)

I believe this is an adaptation of the UCI set, originally provided by Dr. Hans Hofmann of the Universitat at Hamburg, in 2000, found [here](https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)).
