// Copyright (C) 2017 Michael Matthews.
// See the LICENCE.txt file distributed with this work for additional
// information regarding copyright ownership.
//
package ml

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.sql.DataFrame

object LogisticRegressionModel extends BaseModel {

  def run(data: DataFrame): DataFrame = {
    // scalastyle:off magic.number
    val lr = new LogisticRegression()
    val paramGrid = new ParamGridBuilder()
      .addGrid(lr.regParam, Array(0.01, 0.1, 1, 10, 100))
      .build()

    val trainValidationSplit = new TrainValidationSplit()
      .setEstimator(lr)
      .setEvaluator(new BinaryClassificationEvaluator())
      .setEstimatorParamMaps(paramGrid)
      .setTrainRatio(0.8)

    val pipelineBuilder = getStdPipeline()
      .model(trainValidationSplit)

    val Array(training, test) = getStdFeatures(data)

    val pipeline = new Pipeline().setStages(pipelineBuilder.build())
    val model = pipeline.fit(training)
    model.transform(test)
  }
}
