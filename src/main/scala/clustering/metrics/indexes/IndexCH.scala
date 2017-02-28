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

object IndexCH {
  /**
   * **************************************************************************************
   *                   			                                                              *
   * 	  CH INDEX (Calinski and Harabasz) - 1974 	                 					              *
   *    																																									*
   *    Index = (Bq/(q-1)) / (Wq/(n-q))																										*
   *    																					 																				*
   *    The maximum value of the index indicates the best solution												*
   *  								                                  																	*
   * **************************************************************************************
   */
  def calculate(modelTuples: List[TuplaModelos], vectorData: DataFrame) = {
    println(s"CH INDEX -> ${modelTuples.map(_.k)}")

    import Spark.spark.implicits._

    val features = vectorData.rdd.map(x => org.apache.spark.mllib.linalg.Vectors.dense(x.getAs[org.apache.spark.ml.linalg.Vector]("features").toArray))

    val centroideDataSet = Statistics.colStats(features).mean
    val n = vectorData.count()

    val CHIndexesKMeans: ListBuffer[Tuple2[Int, Double]] = ListBuffer[Tuple2[Int, Double]]()
    val CHIndexesBisectingKMeans: ListBuffer[Tuple2[Int, Double]] = ListBuffer[Tuple2[Int, Double]]()
    val CHIndexesGMM: ListBuffer[Tuple3[Int, Double, Int]] = ListBuffer[Tuple3[Int, Double, Int]]()

    for (modelsK <- modelTuples) {
      val k = modelsK.k
      println(s"CALCULANDO CH INDEX PARA k = $k")
      val modelKMeans = modelsK.modelKMeans
      val modelBisectingKMeans = modelsK.modelBisectingKMeans
      val modelGMM = modelsK.modelGMM

      // KMEANS
      if (modelKMeans != null) {
        val clusteredData = modelKMeans._2

        // Within Set Sum of Squared Errors
        val Wq = modelKMeans._1.computeCost(vectorData)

        // Bq = suma de between-group dispersion para cada cluster 
        var Bq = 0.0

        for (cluster <- 0 to k - 1) {
          // nk = numero de objetos en cluster 
          val nk = clusteredData.where("prediction = " + cluster).count
          val centroide = modelKMeans._1.clusterCenters(cluster)
          Bq = Bq + (nk * Vectors.sqdist(centroideDataSet.asML, centroide))
        }

        val CHIndex = (Bq / (k - 1)) / (Wq / (n - k))
        CHIndexesKMeans += Tuple2(k, CHIndex)
      }

      // BISECTING KMEANS
      if (modelBisectingKMeans != null) {
        val clusteredData = modelBisectingKMeans._2

        // Within Set Sum of Squared Errors
        val Wq = modelBisectingKMeans._1.computeCost(vectorData)

        // Bq = suma de between-group dispersion para cada cluster 
        var Bq = 0.0

        for (cluster <- 0 to k - 1) {
          // nk = numero de objetos en cluster 
          val nk = clusteredData.where("prediction = " + cluster).count
          val centroide = modelBisectingKMeans._1.clusterCenters(cluster)
          Bq = Bq + (nk * Vectors.sqdist(centroideDataSet.asML, centroide))
        }

        val CHIndex = (Bq / (k - 1)) / (Wq / (n - k))
        CHIndexesBisectingKMeans += Tuple2(k, CHIndex)
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

        // Bq = suma de between-group dispersion para cada cluster 
        var Bq = 0.0

        for (cluster <- 0 to k - 1) {
          // nk = numero de objetos en cluster 
          val nk = clusteredData.where("prediction = " + cluster).count
          val centroide = modelGMM._1.gaussians(cluster).mean
          Bq = Bq + (nk * Vectors.sqdist(centroideDataSet.asML, centroide))
        }

        val CHIndex = (Bq / (numClustersFinales - 1)) / (Wq / (n - numClustersFinales))
        CHIndexesGMM += Tuple3(numClustersFinales, CHIndex, k)
      }
    }

    val listResultFinal = ListBuffer.empty[ResultIndex]

    if (!CHIndexesKMeans.isEmpty) {
      val result = CHIndexesKMeans.sortBy(x => x._2)
      var points = 0
      for (result_value <- result) {
        listResultFinal += ResultIndex(ClusteringIndexes.METHOD_KMEANS, ClusteringIndexes.INDEX_CH, result_value._1, result_value._2, points, result_value._1)
        points = points + 1
      }
    }

    if (!CHIndexesBisectingKMeans.isEmpty) {
      val result = CHIndexesBisectingKMeans.sortBy(x => x._2)
      var points = 0
      for (result_value <- result) {
        listResultFinal += ResultIndex(ClusteringIndexes.METHOD_BISECTING_KMEANS, ClusteringIndexes.INDEX_CH, result_value._1, result_value._2, points, result_value._1)
        points = points + 1
      }
    }

    if (!CHIndexesGMM.isEmpty) {
      val result = CHIndexesGMM.sortBy(x => x._2)
      var points = 0
      for (result_value <- result) {
        listResultFinal += ResultIndex(ClusteringIndexes.METHOD_GMM, ClusteringIndexes.INDEX_CH, result_value._1, result_value._2, points, result_value._3)
        points = points + 1
      }
    }

    listResultFinal
  }

}