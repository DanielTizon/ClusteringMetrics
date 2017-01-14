package clustering.metrics

import org.apache.spark.sql.Dataset
import org.apache.spark.sql.DataFrame
import org.apache.spark.rdd.RDD

object ExternalValidation {

  /**
   * RAND INDEX = (TP + TN) / (TP + FP + TN + FN)
   */
  def RandIndex(resultClustering: DataFrame, evidenciaAgrupados: RDD[Tuple2[String, String]], evidenciaSeparados: RDD[Tuple2[String, String]]) = {

    val truePositives = Spark.spark.sparkContext.longAccumulator("True Positives")
    val trueNegatives = Spark.spark.sparkContext.longAccumulator("True Negatives")

    val falsePositives = Spark.spark.sparkContext.longAccumulator("False Positives")
    val falseNegatives = Spark.spark.sparkContext.longAccumulator("False Negatives")
    
    evidenciaAgrupados.collect().map(x => {
      val id1 = x._1
      val id2 = x._2

      val grupo1 = resultClustering.filter("id = " + id1).head().getAs[Integer]("prediction")
      val grupo2 = resultClustering.filter("id = " + id2).head().getAs[Integer]("prediction")

      if (grupo1 == grupo2) truePositives.add(1)
      else falseNegatives.add(1)
    })
    
    evidenciaSeparados.map(x => {
      val id1 = x._1
      val id2 = x._2

      val grupo1 = resultClustering.filter("id = " + id1).head().getAs[Integer]("prediction")
      val grupo2 = resultClustering.filter("id = " + id2).head().getAs[Integer]("prediction")

      if (grupo1 != grupo2) trueNegatives.add(1)
      else falsePositives.add(1)
    })

    (truePositives.value + trueNegatives.value) / (truePositives.value + trueNegatives.value + falsePositives.value + falseNegatives.value).toDouble

  }
}