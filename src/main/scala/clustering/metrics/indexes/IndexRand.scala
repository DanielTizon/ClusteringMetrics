package clustering.metrics.indexes

import org.apache.spark.sql.DataFrame
import org.apache.spark.rdd.RDD
import clustering.metrics.Spark

object IndexRand {

  /**
   * ***************************************************************************
   *                   			                                                   *
   * 		RAND INDEX (Krzanowski and Lai) - 1971			   												 *
   * 																																					 *
   * 		Index = (TP + TN) / (TP + FP + TN + FN)																 *
   *     																																			 *
   *    The maximum value of the index indicates the best solution (0-1)     	 *
   *    																							                         *
   *  								                                 												 *
   * ***************************************************************************
   */
  def calculate(resultClustering: DataFrame, evidenciaAgrupados: RDD[Tuple3[Long, String, String]], evidenciaSeparados: RDD[Tuple3[Long, String, String]]) = {

    import Spark.spark.implicits._

    val result = resultClustering.select("ID", "prediction")

    var truePositives = 0L
    var falseNegatives = 0L

    var falsePositives = 0L
    var trueNegatives = 0L

    if (!evidenciaAgrupados.take(1).isEmpty) {

      val dfEvidenciaAgrupadosTotal = evidenciaAgrupados.toDF("ID_GROUP", "ID1", "ID2")
      val dfEvidenciaAgrupados1 = dfEvidenciaAgrupadosTotal.join(result, dfEvidenciaAgrupadosTotal("ID1") === result("ID")).select("ID_GROUP", "prediction").withColumnRenamed("prediction", "prediction_1")
      val dfEvidenciaAgrupados2 = dfEvidenciaAgrupadosTotal.join(result, dfEvidenciaAgrupadosTotal("ID2") === result("ID")).select("ID_GROUP", "prediction").withColumnRenamed("prediction", "prediction_2")
      val joinedAgrupados = dfEvidenciaAgrupados1.join(dfEvidenciaAgrupados2, "ID_GROUP")

      truePositives = joinedAgrupados.where("prediction_1 = prediction_2").count()
      falseNegatives = joinedAgrupados.where("prediction_1 != prediction_2").count()
    }

    if (!evidenciaSeparados.take(1).isEmpty) {

      val dfEvidenciaSeparadosTotal = evidenciaSeparados.toDF("ID_GROUP", "ID1", "ID2")
      val dfEvidenciaSeparados1 = dfEvidenciaSeparadosTotal.join(result, dfEvidenciaSeparadosTotal("ID1") === result("ID")).select("ID_GROUP", "prediction").withColumnRenamed("prediction", "prediction_1")
      val dfEvidenciaSeparados2 = dfEvidenciaSeparadosTotal.join(result, dfEvidenciaSeparadosTotal("ID2") === result("ID")).select("ID_GROUP", "prediction").withColumnRenamed("prediction", "prediction_2")
      val joinedSeparados = dfEvidenciaSeparados1.join(dfEvidenciaSeparados2, "ID_GROUP")

      falsePositives = joinedSeparados.where("prediction_1 = prediction_2").count()
      trueNegatives = joinedSeparados.where("prediction_1 != prediction_2").count()
    }

    (truePositives + trueNegatives) / (truePositives + trueNegatives + falsePositives + falseNegatives).toDouble

  }
}