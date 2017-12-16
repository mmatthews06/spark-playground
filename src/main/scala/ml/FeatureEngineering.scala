// Copyright (C) 2017 Michael Matthews.
// See the LICENCE.txt file distributed with this work for additional
// information regarding copyright ownership.
//
package ml

import org.apache.spark.ml.feature.OneHotEncoder

object FeatureEngineering {
  implicit class StringEngineering(col1: String) {
    def oneHot(col2: String): OneHotEncoder = new OneHotEncoder().setInputCol(col1).setOutputCol(col2)
  }
}