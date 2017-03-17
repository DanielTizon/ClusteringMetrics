package clustering.metrics.indexes

import org.apache.spark.rdd.RDD
import scala.collection.mutable.ListBuffer

import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.rdd.RDD.doubleRDDToDoubleRDDFunctions
import org.apache.spark.sql.DataFrame

import clustering.metrics.Spark
import clustering.metrics.ClusteringIndexes
import clustering.metrics.ClusteringIndexes.ResultIndex
import clustering.metrics.ClusteringIndexes.TuplaModelos

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
  def calculate(modelTuples: List[TuplaModelos], evidencia: DataFrame) = {

    import Spark.spark.implicits._

    println(s"RAND INDEX -> ${modelTuples.map(_.k)}")

    val randIndexesKMeans: ListBuffer[Tuple2[Int, Double]] = ListBuffer[Tuple2[Int, Double]]()
    val randIndexesBisectingKMeans: ListBuffer[Tuple2[Int, Double]] = ListBuffer[Tuple2[Int, Double]]()
    val randIndexesGMM: ListBuffer[Tuple3[Int, Double, Int]] = ListBuffer[Tuple3[Int, Double, Int]]()

    for (modelsK <- modelTuples) {
      val k = modelsK.k
      println(s"CALCULANDO RAND INDEX PARA k = $k")
      val modelKMeans = modelsK.modelKMeans
      val modelBisectingKMeans = modelsK.modelBisectingKMeans
      val modelGMM = modelsK.modelGMM

      // KMEANS
      if (modelKMeans != null) {

        val result = modelKMeans._2.select("ID", "prediction")

        val numCoincidencias = result.join(evidencia, "ID").count()

        if (numCoincidencias > 0) {

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

          val randIndex = (truePositives + trueNegatives) / (truePositives + trueNegatives + falsePositives + falseNegatives).toDouble

          randIndexesKMeans += Tuple2(k, randIndex)
        } else {
          print("En el Dataframe de Evidencias no existe ningún elemento del Dataset")
        }
      }

      // BISECTING KMEANS
      if (modelBisectingKMeans != null) {
        val result = modelBisectingKMeans._2.select("ID", "prediction")

        val numCoincidencias = result.join(evidencia, "ID").count()

        if (numCoincidencias > 0) {

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

          val randIndex = (truePositives + trueNegatives) / (truePositives + trueNegatives + falsePositives + falseNegatives).toDouble

          randIndexesBisectingKMeans += Tuple2(k, randIndex)
        } else {
          print("En el Dataframe de Evidencias no existe ningún elemento del Dataset")
        }
      }

      // MEZCLAS GAUSSIANAS
      if (modelGMM != null) {
        val result = modelGMM._2.select("ID", "prediction")

        val numCoincidencias = result.join(evidencia, "ID").count()

        if (numCoincidencias > 0) {

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

          val randIndex = (truePositives + trueNegatives) / (truePositives + trueNegatives + falsePositives + falseNegatives).toDouble

          randIndexesGMM += Tuple3(k, randIndex, result.select("prediction").distinct.count.toInt)
        } else {
          print("En el Dataframe de Evidencias no existe ningún elemento del Dataset")
        }
      }
    }

    val listResultFinal = ListBuffer.empty[ResultIndex]

    if (!randIndexesKMeans.isEmpty) {
      val result = randIndexesKMeans.sortBy(x => x._2)
      var points = 0
      for (result_value <- result) {
        listResultFinal += ResultIndex(ClusteringIndexes.METHOD_KMEANS, ClusteringIndexes.INDEX_RAND, result_value._1, result_value._2, points, result_value._1)
        points = points + 1
      }
    }

    if (!randIndexesBisectingKMeans.isEmpty) {
      val result = randIndexesKMeans.sortBy(x => x._2)
      var points = 0
      for (result_value <- result) {
        listResultFinal += ResultIndex(ClusteringIndexes.METHOD_BISECTING_KMEANS, ClusteringIndexes.INDEX_RAND, result_value._1, result_value._2, points, result_value._1)
        points = points + 1
      }
    }

    if (!randIndexesGMM.isEmpty) {
      val result = randIndexesGMM.sortBy(x => x._2)
      var points = 0
      for (result_value <- result) {
        listResultFinal += ResultIndex(ClusteringIndexes.METHOD_GMM, ClusteringIndexes.INDEX_RAND, result_value._1, result_value._2, points, result_value._3)
        points = points + 1
      }
    }

    listResultFinal

  }
}