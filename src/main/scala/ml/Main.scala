// Copyright (C) 2017 Michael Matthews.
// See the LICENCE.txt file distributed with this work for additional
// information regarding copyright ownership.
//
package ml

import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.sql.{DataFrame, SparkSession}

object Main {
  Logger.getLogger("org").setLevel(Level.ERROR)

  val spark: SparkSession = SparkSession.builder().getOrCreate()
  spark.sparkContext.setLogLevel("ERROR")
  import spark.implicits._

  def main(args: Array[String] = Array()): Unit = {
    val fileName = if (args.length > 0) args(0) else "./data/german_credit.csv"

    val data = spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .format("csv")
      .load(fileName)

    // Logistic Regression Model
    println("\n" + "-" * 10 + " Logistic Regression " + "-" * 10) // scalastyle:ignore
    val logisticRegressionResults = LogisticRegressionModel.run(data)
    evaluateModel(logisticRegressionResults)

    // Decision Tree Model
    println("\n" + "-" * 10 + " Decision Tree " + "-" * 10) // scalastyle:ignore
    val decisionTreeResults = DecisionTreeModel.run(data)
    evaluateModel(decisionTreeResults)

    // Random Forest Model
    println("\n" + "-" * 10 + " Random Forest " + "-" * 10) // scalastyle:ignore
    val randomForestResults = RandomForestModel.run(data)
    evaluateModel(randomForestResults)

    // PCA run
    println("\n" + "-" * 10 + " Principal Component Analysis " + "-" * 10) // scalastyle:ignore
    val pca = PCAModel.run(data)
    pca.show()

    spark.stop()
  }

  def evaluateModel(results: DataFrame): Unit = {
    // scalastyle:off regex
    val predictionAndLabels = results.select($"prediction", $"label").as[(Double, Double)].rdd
    val metrics = new MulticlassMetrics(predictionAndLabels)

    val accuracy = metrics.accuracy
    println(s"Accuracy = $accuracy")

    println("Confusion matrix:")
    println(metrics.confusionMatrix)

    // Precision by label
    val labels = metrics.labels
    labels.foreach { l =>
      println(s"Precision($l) = " + metrics.precision(l))
    }

    // Recall by label
    labels.foreach { l =>
      println(s"Recall($l) = " + metrics.recall(l))
    }
  }
}
