package clustering.metrics.indexes

import scala.collection.mutable.ListBuffer

import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.mllib.stat.Statistics
import org.apache.spark.rdd.RDD.doubleRDDToDoubleRDDFunctions
import org.apache.spark.sql.DataFrame

import clustering.metrics.Spark
import clustering.metrics.ClusteringIndexes
import clustering.metrics.ClusteringIndexes.ResultIndex
import clustering.metrics.ClusteringIndexes.TuplaModelos

object IndexKL {

  /**
   * ***************************************************************************
   *                   			                                                   *
   * 		KL INDEX (Krzanowski and Lai) - 1988			   													 *
   * 																																					 *
   * 		Index = |DIFFq / DIFFq+1|																							 *
   *     																																			 *
   *    DIFFq = (pow((q - 1), 2/p) * Wq-1) - (pow(q, 2/p) * Wq)                *
   * 																																 					 *
   *    The maximum value of the index indicates the best solution						 *
   *    																							                         *
   *  								                                 												 *
   * ***************************************************************************
   */
  def calculate(modelTuples: List[TuplaModelos], vectorData: DataFrame) = {
    println(s"KL INDEX -> ${modelTuples.map(_.k)}")

    import Spark.spark.implicits._

    val p = vectorData.head().getAs[org.apache.spark.ml.linalg.Vector]("features").size.toDouble

    val WqByKKmeans: ListBuffer[Tuple2[Int, Double]] = ListBuffer[Tuple2[Int, Double]]()
    val WqByKBisectingKmeans: ListBuffer[Tuple2[Int, Double]] = ListBuffer[Tuple2[Int, Double]]()
    val WqByKGMM: ListBuffer[Tuple2[Int, Double]] = ListBuffer[Tuple2[Int, Double]]()

    for (modelsK <- modelTuples if (modelsK.k > 1)) {
      val k = modelsK.k
      println(s"CALCULANDO KL INDEX PARA k = $k")
      val modelKMeans = modelsK.modelKMeans
      val modelBisectingKMeans = modelsK.modelBisectingKMeans
      val modelGMM = modelsK.modelGMM

      // KMEANS
      if (modelKMeans != null) {
        // Wq = WSSSE (Within Set Sum of Squared Errors)
        val Wq = modelKMeans._1.computeCost(vectorData)
        WqByKKmeans += Tuple2(k, Wq)
      }

      // BISECTING KMEANS
      if (modelBisectingKMeans != null) {
        // Wq = WSSSE (Within Set Sum of Squared Errors)
        val Wq = modelBisectingKMeans._1.computeCost(vectorData)
        WqByKBisectingKmeans += Tuple2(k, Wq)
      }

      // MEZCLAS GAUSSIANAS
      if (modelGMM != null) {
        val clusteredData = modelGMM._2
        var numClustersFinales = 0
        var Wq = 0.0
        for (cluster <- 0 to k - 1) {
          val clusterData = clusteredData.where("prediction =" + cluster)
          val numObjetosCluster = clusterData.count()
          if (numObjetosCluster > 0) {
            numClustersFinales = numClustersFinales + 1
            val centroide = modelGMM._1.gaussians(cluster).mean
            Wq = Wq + clusterData.map(x => Vectors.sqdist(centroide, x.getAs[org.apache.spark.ml.linalg.Vector]("features"))).rdd.sum
          }
        }
        WqByKGMM += Tuple2(numClustersFinales, Wq)
      }
    }

    val listResultFinal = ListBuffer.empty[ResultIndex]

    if (!WqByKKmeans.isEmpty) {
      if (modelTuples.map(_.k).sorted.head == 1) {
        val features = vectorData.rdd.map(x => org.apache.spark.mllib.linalg.Vectors.dense(x.getAs[org.apache.spark.ml.linalg.Vector]("features").toArray))
        val centroideDataSet = Statistics.colStats(features).mean
        val Wq = vectorData.map(x => Vectors.sqdist(centroideDataSet.asML, x.getAs[org.apache.spark.ml.linalg.Vector]("features"))).rdd.sum()
        WqByKKmeans += Tuple2(1, Wq)
      }

      val DIFFsq = WqByKKmeans.sortBy(x => x._1).sliding(2).map(x => {
        val qActual = x(1)._1
        val WqActual = x(1)._2
        val qAnterior = x(0)._1
        val WqAnterior = x(0)._2

        val resAnterior = (math.pow((qAnterior), (2 / p))) * WqAnterior
        val resActual = (math.pow(qActual, (2 / p))) * WqActual
        val DIFFq = resAnterior - resActual
        (qActual, DIFFq)
      }).toList

      val KLIndexes = DIFFsq.sortBy(_._1).sliding(2).map(x => {
        val DIFFq1 = x(0)._2
        val DIFFq2 = x(1)._2
        val KLIndex = math.abs(DIFFq1 / DIFFq2)
        (x(0)._1, KLIndex)
      }).toList

      val result = KLIndexes.sortBy(x => x._2)
      var points = 0
      for (result_value <- result) {
        listResultFinal += ResultIndex(ClusteringIndexes.METHOD_KMEANS, ClusteringIndexes.INDEX_KL, result_value._1, result_value._2, points)
      }
    }

    if (!WqByKBisectingKmeans.isEmpty) {
      if (modelTuples.map(_.k).sorted.head == 1) {
        val features = vectorData.rdd.map(x => org.apache.spark.mllib.linalg.Vectors.dense(x.getAs[org.apache.spark.ml.linalg.Vector]("features").toArray))
        val centroideDataSet = Statistics.colStats(features).mean
        val Wq = vectorData.map(x => Vectors.sqdist(centroideDataSet.asML, x.getAs[org.apache.spark.ml.linalg.Vector]("features"))).rdd.sum()
        WqByKBisectingKmeans += Tuple2(1, Wq)
      }

      val DIFFsq = WqByKBisectingKmeans.sortBy(x => x._1).sliding(2).map(x => {
        val qActual = x(1)._1
        val WqActual = x(1)._2
        val qAnterior = x(0)._1
        val WqAnterior = x(0)._2

        val resAnterior = (math.pow((qAnterior), (2 / p))) * WqAnterior
        val resActual = (math.pow(qActual, (2 / p))) * WqActual
        val DIFFq = resAnterior - resActual
        (qActual, DIFFq)
      }).toList

      val KLIndexes = DIFFsq.sortBy(x => x._1).sliding(2).map(x => {
        val DIFFq1 = x(0)._2
        val DIFFq2 = x(1)._2
        val KLIndex = math.abs(DIFFq1 / DIFFq2)
        (x(0)._1, KLIndex)
      }).toList

      val result = KLIndexes.sortBy(x => x._2)
      var points = 0
      for (result_value <- result) {
        listResultFinal += ResultIndex(ClusteringIndexes.METHOD_BISECTING_KMEANS, ClusteringIndexes.INDEX_KL, result_value._1, result_value._2, points)
      }
    }

    if (!WqByKGMM.isEmpty) {
      if (modelTuples.map(_.k).sorted.head == 1) {
        val features = vectorData.rdd.map(x => org.apache.spark.mllib.linalg.Vectors.dense(x.getAs[org.apache.spark.ml.linalg.Vector]("features").toArray))
        val centroideDataSet = Statistics.colStats(features).mean
        val Wq = vectorData.map(x => Vectors.sqdist(centroideDataSet.asML, x.getAs[org.apache.spark.ml.linalg.Vector]("features"))).rdd.sum()
        WqByKGMM += Tuple2(1, Wq)
      }

      val DIFFsq = WqByKGMM.sortBy(x => x._1).sliding(2).map(x => {
        val qActual = x(1)._1
        val WqActual = x(1)._2
        val qAnterior = x(0)._1
        val WqAnterior = x(0)._2

        val resAnterior = (math.pow((qAnterior), (2 / p))) * WqAnterior
        val resActual = (math.pow(qActual, (2 / p))) * WqActual
        val DIFFq = resAnterior - resActual
        (qActual, DIFFq)
      }).toList

      val KLIndexes = DIFFsq.sortBy(x => x._1).sliding(2).map(x => {
        val DIFFq1 = x(0)._2
        val DIFFq2 = x(1)._2
        val KLIndex = math.abs(DIFFq1 / DIFFq2)
        (x(0)._1, KLIndex)
      }).toList

      val result = KLIndexes.sortBy(x => x._2)
      var points = 0
      for (result_value <- result) {
        listResultFinal += ResultIndex(ClusteringIndexes.METHOD_GMM, ClusteringIndexes.INDEX_KL, result_value._1, result_value._2, points)
      }
    }

    listResultFinal

  }
}