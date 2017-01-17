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
    val WqByKGMM: ListBuffer[Tuple2[Int, Double]] = ListBuffer[Tuple2[Int, Double]]()

    for (modelsK <- modelTuples) {
      val k = modelsK.k
      val modelKMeans = modelsK.modelKMeans
      val modelBisectingKMeans = modelsK.modelBisectingKMeans
      val modelGMM = modelsK.modelGMM

      // KMEANS
      if (modelKMeans != null) {
        // Wq = WSSSE (Within Set Sum of Squared Errors)
        val Wq = modelKMeans.computeCost(vectorData)
        WqByKKmeans += Tuple2(k, Wq)
      }

      // BISECTING KMEANS
      if (modelBisectingKMeans != null) {
        // Wq = WSSSE (Within Set Sum of Squared Errors)
        val Wq = modelBisectingKMeans.computeCost(vectorData)
        WqByKBisectingKmeans += Tuple2(k, Wq)
      }

      // MEZCLAS GAUSSIANAS
      if (modelGMM != null) {
        val clusteredData = modelGMM.transform(vectorData)
        var numClustersFinales = 0
        var Wq = 0.0
        for (cluster <- 0 to k - 1) {
          val clusterData = clusteredData.where("prediction = "+cluster)
          val numObjetosCluster = clusterData.count()
          if (numObjetosCluster > 0) {
            numClustersFinales = numClustersFinales + 1
            val centroide = modelGMM.gaussians(cluster).mean
            Wq = Wq + clusterData.map(x => Vectors.sqdist(centroide, x.getAs[org.apache.spark.ml.linalg.Vector]("features"))).rdd.sum
          }
        }
        WqByKGMM += Tuple2(numClustersFinales, Wq)
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

      val result = listaSlopes.sortBy(x => x._2).last
      listResultFinal += ResultIndex(ClusteringIndexes.METHOD_KMEANS, ClusteringIndexes.INDEX_HARTIGAN, result._1, result._2)
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

      val result = listaSlopes.sortBy(x => x._2).last
      listResultFinal += ResultIndex(ClusteringIndexes.METHOD_BISECTING_KMEANS, ClusteringIndexes.INDEX_HARTIGAN, result._1, result._2)
    }

    if (!WqByKGMM.isEmpty) {
      val hartiganIndexes = WqByKGMM.sortBy(x => x._1).sliding(2).map(x => {
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

      val result = listaSlopes.sortBy(x => x._2).last
      listResultFinal += ResultIndex(ClusteringIndexes.METHOD_GMM, ClusteringIndexes.INDEX_HARTIGAN, result._1, result._2)
    }

    listResultFinal

  }
}