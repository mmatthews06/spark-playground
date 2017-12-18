// Copyright (C) 2017 Michael Matthews.
// See the LICENCE.txt file distributed with this work for additional
// information regarding copyright ownership.
//
package ml

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.sql.DataFrame

object DecisionTreeModel extends BaseModel {

  def run(data: DataFrame): DataFrame = {
    // scalastyle:off magic.number
    val classifier = new DecisionTreeClassifier()
    val paramGrid = new ParamGridBuilder()
      .addGrid(classifier.maxDepth, Array(1, 3, 5, 10, 15, 20, 25))
      .build()

    val trainValidationSplit = new TrainValidationSplit()
      .setEstimator(classifier)
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
