// Copyright (C) 2017 Michael Matthews.
// See the LICENCE.txt file distributed with this work for additional
// information regarding copyright ownership.
//
package ml

import ml.FeatureEngineering._
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.VectorAssembler
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
    logisticRegression(data)

    // Decision Tree Model
    println("\n" + "-" * 10 + " Decision Tree " + "-" * 10) // scalastyle:ignore
    val decisionTreeResults = DecisionTreeModel.run(data)
    evaluateModel(decisionTreeResults)

    // PCA run
    println("\n" + "-" * 10 + " Principal Component Analysis " + "-" * 10) // scalastyle:ignore
    val pca = PCAModel.run(data)
    pca.show()

    spark.stop()
  }

  def logisticRegression(data: DataFrame) {
    val seed = 42
    val trainSplit = 0.70
    val testSplit = 0.30

    val balanceEncoder = "Account Balance" oneHot "balanceOH"
    val guarantorEncoder = "Guarantors" oneHot "guarantorsOH"
    val paymentStatusEncoder = "Payment Status of Previous Credit" oneHot "paymentStatusOH"
    val maritalStatusEncoder = "Sex & Marital Status" oneHot "maritalStatusOH"
    val currentCreditEncoder = "Concurrent Credits" oneHot "currentCreditOH"

    val featuresAndLabels = data.select(
      data("Creditability").as("label"), $"Account Balance", $"Duration of Credit (month)",
      $"Payment Status of Previous Credit", $"Credit Amount", $"Sex & Marital Status", $"Guarantors", $"Age (years)",
      $"Concurrent Credits"
    )

    val cleanData = featuresAndLabels.na.drop()
    val Array(training, test) = cleanData.randomSplit(Array(trainSplit, testSplit), seed=seed)

    val featureExtractor = new VectorAssembler()
        .setInputCols(Array(
          "balanceOH", "paymentStatusOH", "maritalStatusOH", "guarantorsOH", "currentCreditOH",
          "Duration of Credit (month)", "Credit Amount", "Age (years)"
        ))
        .setOutputCol("features")

    val lr = new LogisticRegression()
    val pipeline = new Pipeline().setStages(Array(
      balanceEncoder,
      guarantorEncoder,
      paymentStatusEncoder,
      maritalStatusEncoder,
      currentCreditEncoder,
      featureExtractor,
      lr
    ))

    val model = pipeline.fit(training)
    val results = model.transform(test)

    evaluateModel(results)
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
