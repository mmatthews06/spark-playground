// Copyright (C) 2017 Michael Matthews.
// See the LICENCE.txt file distributed with this work for additional
// information regarding copyright ownership.
//
package ml

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.sql.{DataFrame, SparkSession}

object RandomForestModel {
  val spark: SparkSession = SparkSession.builder().getOrCreate()
  import spark.implicits._

  def run(data: DataFrame): DataFrame = {
    val seed = 42
    val trainSplit = 0.70
    val testSplit = 0.30

    val classifier = new RandomForestClassifier()
    val pipelineBuilder = new PipelineBuilder()
      .oneHot("Account Balance", "balanceOH")
      .oneHot("Guarantors", "guarantorsOH")
      .oneHot("Payment Status of Previous Credit", "paymentStatusOH")
      .oneHot("Sex & Marital Status", "maritalStatusOH")
      .oneHot("Concurrent Credits", "currentCreditOH")
      .features(
        "balanceOH",
        "paymentStatusOH",
        "maritalStatusOH",
        "guarantorsOH",
        "currentCreditOH",
        "Duration of Credit (month)",
        "Credit Amount",
        "Age (years)"
      )
      .model(classifier)

    val Array(training, test)  = data.select(
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
      .na.drop()
      .randomSplit(Array(trainSplit, testSplit), seed=seed)

    val pipeline = new Pipeline().setStages(pipelineBuilder.build())
    val model = pipeline.fit(training)
    model.transform(test)
  }
}
