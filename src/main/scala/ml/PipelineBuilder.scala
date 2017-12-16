// Copyright (C) 2017 Michael Matthews.
// See the LICENCE.txt file distributed with this work for additional
// information regarding copyright ownership.
//

package ml

import ml.FeatureEngineering._
import org.apache.spark.ml.PipelineStage
import org.apache.spark.ml.feature.VectorAssembler


class PipelineBuilder(pipeline: List[PipelineStage] = List[PipelineStage]()) {

  def model(model: PipelineStage): PipelineBuilder = {
    val newPipeline = pipeline :+ model
    new PipelineBuilder(newPipeline)
  }

  def oneHot(col1: String, col2:String): PipelineBuilder = {
    val encoder = col1 oneHot col2
    val newPipeline = pipeline :+ encoder
    new PipelineBuilder(newPipeline)
  }

  def features(features: String*): PipelineBuilder = {
    val featureExtractor = new VectorAssembler()
      .setInputCols(features.toArray)
      .setOutputCol("features")

    val newPipeline = pipeline :+ featureExtractor
    new PipelineBuilder(newPipeline)
  }

  def build(): Array[PipelineStage] = {
    pipeline.toArray
  }
}
