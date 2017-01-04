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
    val CHIndexesGMM: ListBuffer[Tuple2[Int, Double]] = ListBuffer[Tuple2[Int, Double]]()

    for (modelsK <- modelTuples) {
      val k = modelsK.k
      val modelKMeans = modelsK.modelKMeans
      val modelBisectingKMeans = modelsK.modelBisectingKMeans
      val modelGMM = modelsK.modelGMM

      // KMEANS
      if (modelKMeans != null) {
        val clusteredData = modelKMeans.transform(vectorData)

        // Within Set Sum of Squared Errors
        val Wq = modelKMeans.computeCost(vectorData)

        // Bq = suma de between-group dispersion para cada cluster 
        var Bq = 0.0

        for (cluster <- 0 to k - 1) {
          // nk = numero de objetos en cluster 
          val nk = clusteredData.where("prediction = "+cluster).count
          val centroide = modelKMeans.clusterCenters(cluster)
          Bq = Bq + (nk * Vectors.sqdist(centroideDataSet.asML, centroide))
        }

        val CHIndex = (Bq / (k - 1)) / (Wq / (n - k))
        CHIndexesKMeans += Tuple2(k, CHIndex)
      }

      // BISECTING KMEANS
      if (modelBisectingKMeans != null) {
        val clusteredData = modelBisectingKMeans.transform(vectorData)

        // Within Set Sum of Squared Errors
        val Wq = modelBisectingKMeans.computeCost(vectorData)

        // Bq = suma de between-group dispersion para cada cluster 
        var Bq = 0.0

        for (cluster <- 0 to k - 1) {
          // nk = numero de objetos en cluster 
          val nk = clusteredData.where("prediction = "+ cluster).count
          val centroide = modelBisectingKMeans.clusterCenters(cluster)
          Bq = Bq + (nk * Vectors.sqdist(centroideDataSet.asML, centroide))
        }

        val CHIndex = (Bq / (k - 1)) / (Wq / (n - k))
        CHIndexesBisectingKMeans += Tuple2(k, CHIndex)
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

        // Bq = suma de between-group dispersion para cada cluster 
        var Bq = 0.0

        for (cluster <- 0 to k - 1) {
          // nk = numero de objetos en cluster 
          val nk = clusteredData.where("prediction = "+cluster).count
          val centroide = modelGMM.gaussians(cluster).mean
          Bq = Bq + (nk * Vectors.sqdist(centroideDataSet.asML, centroide))
        }

        val CHIndex = (Bq / (numClustersFinales - 1)) / (Wq / (n - numClustersFinales))
        CHIndexesGMM += Tuple2(numClustersFinales, CHIndex)
      }
    }

    val listResultFinal = ListBuffer.empty[ResultIndex]

    if (!CHIndexesKMeans.isEmpty) {
      val result = CHIndexesKMeans.sortBy(x => x._2).last
      listResultFinal += ResultIndex(ClusteringIndexes.METHOD_KMEANS, ClusteringIndexes.INDEX_CH, result._1, result._2)
    }

    if (!CHIndexesBisectingKMeans.isEmpty) {
      val result = CHIndexesBisectingKMeans.sortBy(x => x._2).last
      listResultFinal += ResultIndex(ClusteringIndexes.METHOD_BISECTING_KMEANS, ClusteringIndexes.INDEX_CH, result._1, result._2)
    }

    if (!CHIndexesGMM.isEmpty) {
      val result = CHIndexesGMM.sortBy(x => x._2).last
      listResultFinal += ResultIndex(ClusteringIndexes.METHOD_GMM, ClusteringIndexes.INDEX_CH, result._1, result._2)
    }

    listResultFinal
  }

}