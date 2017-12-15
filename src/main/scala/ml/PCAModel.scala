// Copyright (C) 2017 Michael Matthews.
// See the LICENCE.txt file distributed with this work for additional
// information regarding copyright ownership.
//
package ml

import org.apache.spark.ml.feature.{PCA, VectorAssembler}
import org.apache.spark.sql.{DataFrame, SparkSession}

object PCAModel {
  val spark: SparkSession = SparkSession.builder().getOrCreate()
  import spark.implicits._

  def run(data: DataFrame): DataFrame = {
    val K = 3
    val featuresAndLabels = data.select(
      data("Creditability").as("label"), $"Account Balance", $"Duration of Credit (month)",
      $"Payment Status of Previous Credit", $"Credit Amount", $"Sex & Marital Status", $"Guarantors", $"Age (years)",
      $"Concurrent Credits"
    )

    val cleanData = featuresAndLabels.na.drop()

    val featureExtractor = new VectorAssembler()
      .setInputCols(Array(
        "Account Balance", "Payment Status of Previous Credit", "Sex & Marital Status", "Guarantors",
        "Concurrent Credits", "Duration of Credit (month)", "Credit Amount", "Age (years)"
      ))
      .setOutputCol("features")

    val featureData = featureExtractor.transform(cleanData).select($"features")

    val pca = new PCA()
      .setInputCol("features")
      .setOutputCol("pca")
      .setK(K)
      .fit(featureData)

    pca.transform(featureData).select("pca")
  }
}
