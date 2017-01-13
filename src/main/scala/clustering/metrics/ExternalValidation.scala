package clustering.metrics

import org.apache.spark.sql.Dataset
import org.apache.spark.sql.DataFrame

object ExternalValidation {

  /**
   * RAND INDEX = (TP + TN) / (TP + FP + TN + FN)
   */
  def RandIndexKMeans(resultKMeans: DataFrame, evidenciaAgrupados: List[Tuple2[String, String]], evidenciaSeparados: List[Tuple2[String, String]]) = {

    val truePositives = Spark.spark.sparkContext.longAccumulator("True Positives")
    val trueNegatives = Spark.spark.sparkContext.longAccumulator("True Negatives")

    val falsePositives = Spark.spark.sparkContext.longAccumulator("False Positives")
    val falseNegatives = Spark.spark.sparkContext.longAccumulator("False Negatives")

    evidenciaAgrupados.map(x => {
      val tycId1 = x._1
      val tycId2 = x._2

      val grupo1 = resultKMeans.filter("id = " + tycId1).head().getAs[Double]("prediction")
      val grupo2 = resultKMeans.filter("id = " + tycId2).head().getAs[Double]("prediction")

      if (grupo1 == grupo2) truePositives.add(1)
      else falseNegatives.add(1)
    })

    evidenciaSeparados.map(x => {
      val tycId1 = x._1
      val tycId2 = x._2

      val grupo1 = resultKMeans.filter("id = " + tycId1).head().getAs[Double]("prediction")
      val grupo2 = resultKMeans.filter("id = " + tycId2).head().getAs[Double]("prediction")

      if (grupo1 != grupo2) trueNegatives.add(1)
      else falsePositives.add(1)
    })

    (truePositives.value + trueNegatives.value) / (truePositives.value + trueNegatives.value + falsePositives.value + falseNegatives.value).toDouble

  }

  def RandIndexGMM(resultKMeans: DataFrame, evidenciaAgrupados: List[Tuple2[String, String]], evidenciaSeparados: List[Tuple2[String, String]]) = {

    val truePositives = Spark.spark.sparkContext.longAccumulator("True Positives")
    val trueNegatives = Spark.spark.sparkContext.longAccumulator("True Negatives")

    val falsePositives = Spark.spark.sparkContext.longAccumulator("False Positives")
    val falseNegatives = Spark.spark.sparkContext.longAccumulator("False Negatives")

    evidenciaAgrupados.map(x => {
      val tycId1 = x._1
      val tycId2 = x._2

      val grupo1 = resultKMeans.filter("id = " + tycId1).head().getAs[Double]("prediction")
      val grupo2 = resultKMeans.filter("id = " + tycId1).head().getAs[Double]("prediction")

      if (grupo1 == grupo2) truePositives.add(1)
      else falseNegatives.add(1)
    })

    evidenciaSeparados.map(x => {
      val tycId1 = x._1
      val tycId2 = x._2

      val grupo1 = resultKMeans.filter("id = " + tycId1).head().getAs[Double]("prediction")
      val grupo2 = resultKMeans.filter("id = " + tycId2).head().getAs[Double]("prediction")

      if (grupo1 != grupo2) trueNegatives.add(1)
      else falsePositives.add(1)
    })

    (truePositives.value + trueNegatives.value) / (truePositives.value + trueNegatives.value + falsePositives.value + falseNegatives.value).toDouble
  }
}