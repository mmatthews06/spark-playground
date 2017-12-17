// Copyright (C) 2017 Michael Matthews.
// See the LICENCE.txt file distributed with this work for additional
// information regarding copyright ownership.
//

package ml

import ml.FeatureEngineering._
import org.apache.spark.ml.PipelineStage

class PipelineBuilder(pipeline: List[PipelineStage] = List[PipelineStage]()) {

  def build(): Array[PipelineStage] = pipeline.toArray

  def features(features: String*): PipelineBuilder = new PipelineBuilder(pipeline :+ features.extract)

  def model(model: PipelineStage): PipelineBuilder = new PipelineBuilder(pipeline :+ model)

  def oneHot(col1: String, col2:String): PipelineBuilder = new PipelineBuilder(pipeline :+ (col1 oneHot col2))

}
