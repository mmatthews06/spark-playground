// Copyright (C) 2017 Michael Matthews.
// See the LICENCE.txt file distributed with this work for additional
// information regarding copyright ownership.
//

import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.mllib.evaluation.MulticlassMetrics

object Main {

  val spark: SparkSession = SparkSession.builder().getOrCreate()
  import spark.implicits._

  def main(args: Array[String] = Array()): Unit = {
    val fileName = if (args.length > 0) args(0) else "./data/german_credit.csv"

    val data = spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .format("csv")
      .load(fileName)

    logisticRegression(data)

    spark.stop()
  }

  def logisticRegression(data: DataFrame) {
    // Really poor steps for logistic regression.
    val seed = 42
    val trainSplit = 0.70
    val testSplit = 0.30
    val featuresAndLabels = data.select(
      data("Creditability").as("label"),
      $"Account Balance",
      $"Duration of Credit (month)",
      $"Payment Status of Previous Credit",
      $"Credit Amount",
      $"Sex & Marital Status",
      $"Guarantors",
      $"Age (years)",
      $"Concurrent Credits"
    )

    val cleanData = featuresAndLabels.na.drop()

    val featureExtractor = new VectorAssembler()
        .setInputCols(Array(
          "Account Balance",
          "Duration of Credit (month)",
          "Payment Status of Previous Credit",
          "Credit Amount",
          "Sex & Marital Status",
          "Guarantors",
          "Age (years)",
          "Concurrent Credits")
        )
        .setOutputCol("features")

    val Array(training, test) = cleanData.randomSplit(Array(trainSplit, testSplit), seed=seed)

    val lr = new LogisticRegression()
    val pipeline = new Pipeline().setStages(Array(featureExtractor, lr))

    val model = pipeline.fit(training)
    val results = model.transform(test)

    evaluateModel(test, results)
  }

  def evaluateModel(test: DataFrame, results: DataFrame): Unit = {
    // scalastyle:off regex
    val predictionAndLabels = results.select($"prediction",$"label").as[(Double, Double)].rdd
    val metrics = new MulticlassMetrics(predictionAndLabels)

    // Overall Statistics
    val accuracy = metrics.accuracy
    println("Summary Statistics")
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
