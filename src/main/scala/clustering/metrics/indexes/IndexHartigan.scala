package clustering.metrics.indexes

import scala.collection.mutable.ListBuffer

import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.rdd.RDD.doubleRDDToDoubleRDDFunctions
import org.apache.spark.sql.DataFrame

import clustering.metrics.Spark
import clustering.metrics.ClusteringIndexes
import clustering.metrics.ClusteringIndexes.ResultIndex
import clustering.metrics.ClusteringIndexes.TuplaModelos

object IndexHartigan {
  /**
   * **************************************************************************
   *                   			                                                  *
   * 	  HARTIGAN INDEX (Hartigan) - 1975																			*
   * 																																					*
   *    Index = ((Wq / Wq+1) - 1) * (n - q - 1)          					            *
   *    																							 												*
   *    The maximum difference between levels indicates the best solution			*
   *  								                                  											*
   * **************************************************************************
   */
  def calculate(modelTuples: List[TuplaModelos], vectorData: DataFrame) = {
    println(s"HARTIGAN INDEX -> ${modelTuples.map(_.k)}")

    import Spark.spark.implicits._

    val n = vectorData.count()
    val WqByKKmeans: ListBuffer[Tuple2[Int, Double]] = ListBuffer[Tuple2[Int, Double]]()
    val WqByKBisectingKmeans: ListBuffer[Tuple2[Int, Double]] = ListBuffer[Tuple2[Int, Double]]()
    val WqByKGMM: ListBuffer[Tuple3[Int, Double, Int]] = ListBuffer[Tuple3[Int, Double, Int]]()

    for (modelsK <- modelTuples) {
      val k = modelsK.k
      println(s"CALCULANDO HARTIGAN INDEX PARA k = $k")
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
          val clusterData = clusteredData.where("prediction = " + cluster)
          val numObjetosCluster = clusterData.count()
          if (numObjetosCluster > 0) {
            numClustersFinales = numClustersFinales + 1
            Wq = Wq + clusterData.map(x => x.getAs[Double]("MaxProb")).rdd.sum
          }
        }
        WqByKGMM += Tuple3(numClustersFinales, Wq, k)
      }
    }

    val listResultFinal = ListBuffer.empty[ResultIndex]

    if (!WqByKKmeans.isEmpty) {
      val hartiganIndexes = WqByKKmeans.sortBy(x => x._1).sliding(2).map(x => {
        val Wq1 = x(0)._2
        val k1 = x(0)._1
        val Wq2 = x(1)._2
        val hartiganIndex = ((Wq1 / Wq2) - 1) * (n - k1 - 1)
        (k1, hartiganIndex)
      }).toList

      val listaSlopes = hartiganIndexes.sortBy(x => x._1).sliding(2).map(x => {
        val hartiganIndex1 = x(0)
        val hartiganIndex2 = x(1)
        (hartiganIndex2._1, Math.abs(hartiganIndex1._2 - hartiganIndex2._2))
      }).toList

      val result = listaSlopes.sortBy(x => x._2)
      var points = 0
      for (result_value <- result) {
        listResultFinal += ResultIndex(ClusteringIndexes.METHOD_KMEANS, ClusteringIndexes.INDEX_HARTIGAN, result_value._1, result_value._2, points, result_value._1)
        points = points + 1
      }
    }

    if (!WqByKBisectingKmeans.isEmpty) {
      val hartiganIndexes = WqByKBisectingKmeans.sortBy(x => x._1).sliding(2).map(x => {
        val Wq1 = x(0)._2
        val k1 = x(0)._1
        val Wq2 = x(1)._2
        val hartiganIndex = ((Wq1 / Wq2) - 1) * (n - k1 - 1)
        (k1, hartiganIndex)
      }).toList

      val listaSlopes = hartiganIndexes.sortBy(x => x._1).sliding(2).map(x => {
        val hartiganIndex1 = x(0)
        val hartiganIndex2 = x(1)
        (hartiganIndex2._1, Math.abs(hartiganIndex1._2 - hartiganIndex2._2))
      }).toList

      val result = listaSlopes.sortBy(x => x._2)
      var points = 0
      for (result_value <- result) {
        listResultFinal += ResultIndex(ClusteringIndexes.METHOD_BISECTING_KMEANS, ClusteringIndexes.INDEX_HARTIGAN, result_value._1, result_value._2, points, result_value._1)
        points = points + 1
      }
    }

    if (!WqByKGMM.isEmpty) {
      val hartiganIndexes = WqByKGMM.sortBy(x => x._1).sliding(2).map(x => {
        val Wq1 = x(0)._2
        val k1 = x(0)._1
        val Wq2 = x(1)._2
        val hartiganIndex = ((Wq1 / Wq2) - 1) * (n - k1 - 1)
        (k1, hartiganIndex, x(0)._3)
      }).toList

      val listaSlopes = hartiganIndexes.sortBy(x => x._1).sliding(2).map(x => {
        val hartiganIndex1 = x(0)
        val hartiganIndex2 = x(1)
        (hartiganIndex2._1, Math.abs(hartiganIndex1._2 - hartiganIndex2._2), hartiganIndex2._3)
      }).toList

      val result = listaSlopes.sortBy(x => x._2)
      var points = 0
      for (result_value <- result) {
        listResultFinal += ResultIndex(ClusteringIndexes.METHOD_GMM, ClusteringIndexes.INDEX_HARTIGAN, result_value._1, result_value._2, points, result_value._3)
        points = points + 1
      }
    }

    listResultFinal

  }
}