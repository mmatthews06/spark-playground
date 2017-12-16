// Copyright (C) 2017 Michael Matthews.
// See the LICENCE.txt file distributed with this work for additional
// information regarding copyright ownership.
//
package ml

import ml.FeatureEngineering._
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.{DataFrame, SparkSession}

object RandomForestModel {
  val spark: SparkSession = SparkSession.builder().getOrCreate()
  import spark.implicits._

  def run(data: DataFrame): DataFrame = {
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

    val decisionTree = new RandomForestClassifier()
    val pipeline = new Pipeline().setStages(Array(
      balanceEncoder,
      guarantorEncoder,
      paymentStatusEncoder,
      maritalStatusEncoder,
      currentCreditEncoder,
      featureExtractor,
      decisionTree
    ))

    val model = pipeline.fit(training)
    model.transform(test)
  }
}
