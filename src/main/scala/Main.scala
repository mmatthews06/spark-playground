
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.VectorAssembler

object Main {

  def main(args: Array[String] = Array()): Unit = {
    val fileName = if (args.length > 0) args(0) else "./data/german_credit.csv";
    val spark = SparkSession.builder().getOrCreate()

    val data = spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .format("csv")
      .load(fileName)

    logisticRegression(data)

    spark.stop()
  }

  def logisticRegression(data: DataFrame) {
    // Really poor steps for logistic regression.
    val featuresAndLabels = data.select(
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

    val cleanData = featuresAndLabels.na.drop()

    val featureExtractor = new VectorAssembler()
        .setInputCols(Array(
          "Account Balance",
          "Duration of Credit (month)",
          "Payment Status of Previous Credit",
          "Credit Amount",
          "Sex & Marital Status",
          "Guarantors",
          "Age (years)",
          "Concurrent Credits")
        )
        .setOutputCol("features")

    val Array(training, test) = cleanData.randomSplit(Array(0.70, 0.30), seed=42)

    val lr = new LogisticRegression()
    val pipeline = new Pipeline().setStages(Array(featureExtractor, lr))

    val model = pipeline.fit(training)
    val results = model.transform(test)

    results.select("features", "label", "prediction").show()
  }
}
