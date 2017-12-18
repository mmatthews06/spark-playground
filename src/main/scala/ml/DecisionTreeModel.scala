// Copyright (C) 2017 Michael Matthews.
// See the LICENCE.txt file distributed with this work for additional
// information regarding copyright ownership.
//
package ml

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.sql.DataFrame

object DecisionTreeModel extends BaseModel {

  def run(data: DataFrame): DataFrame = {
    val classifier = new DecisionTreeClassifier()

    val pipelineBuilder = getStdPipeline()
      .model(classifier)

    val Array(training, test) = getStdFeatures(data)

    val pipeline = new Pipeline().setStages(pipelineBuilder.build())
    val model = pipeline.fit(training)
    model.transform(test)
  }

}
