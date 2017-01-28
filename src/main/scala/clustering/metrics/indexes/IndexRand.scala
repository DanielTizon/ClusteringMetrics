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
  def calculate(resultClustering: DataFrame, evidencia: DataFrame) = {

    import Spark.spark.implicits._

    val result = resultClustering.select("ID", "prediction")

    val evidenciaRDD = evidencia.filter("ID is not null and GRUPO is not null").rdd.map(x => (x.getAs[String]("GRUPO"), x.getAs[String]("ID")))
    val evidenciaAgrupados: RDD[Tuple3[Long, String, String]] = evidenciaRDD.groupByKey.flatMap(x => x._2.toSet.subsets(2)).map(x => (x.head, x.last)).zipWithIndex().map(x => (x._2, x._1._1, x._1._2))
    val evidenciaSeparados: RDD[Tuple3[Long, String, String]] = evidenciaRDD.cartesian(evidenciaRDD).filter(x => x._1._1 != x._2._1).map(x => (x._1._2, x._2._2)).zipWithIndex().map(x => (x._2, x._1._1, x._1._2))

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