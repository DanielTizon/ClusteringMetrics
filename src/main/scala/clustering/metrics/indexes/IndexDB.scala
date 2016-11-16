package clustering.metrics.indexes

import scala.collection.mutable.ListBuffer
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.Dataset
import clustering.metrics.Results.ResultGMM
import clustering.metrics.Results.ResultKMeans
import clustering.metrics.Results.VectorData
import clustering.metrics.ClusteringIndexes
import org.apache.spark.rdd.RDD.doubleRDDToDoubleRDDFunctions
import clustering.metrics.Results.ResultIndex
import clustering.metrics.Results.TuplaModelos
import clustering.metrics.Spark

object IndexDB {
  /**
   * ********************************************************************
   *                   			                                            *
   * 	  DB INDEX (Davies and Boulding) - 1979 	 												*
   * 																																		*
   *    Index = 1/q * SUMk max(k!=l) (dk+dl)/dkl                 				*
   *    																							 									*
   *    The minimum value of the index indicate the best solution				*
   *  								                                  								*
   * ********************************************************************
   */
  def calculate(modelTuples: List[TuplaModelos], vectorData: Dataset[VectorData]) = {
    println(s"DB INDEX -> ${modelTuples.map(_.k)}")

    import Spark.spark.implicits._

    val DBIndexesKMeans: ListBuffer[Tuple2[Int, Double]] = ListBuffer[Tuple2[Int, Double]]()
    val DBIndexesBisectingKMeans: ListBuffer[Tuple2[Int, Double]] = ListBuffer[Tuple2[Int, Double]]()
    val DBIndexesGMM: ListBuffer[Tuple2[Int, Double]] = ListBuffer[Tuple2[Int, Double]]()

    for (modelsK <- modelTuples) {
      val k = modelsK.k
      val modelKMeans = modelsK.modelKMeans
      val modelBisectingKMeans = modelsK.modelBisectingKMeans
      val modelGMM = modelsK.modelGMM

      // KMEANS
      if (modelKMeans != null) {
        val clusteredData = modelKMeans.transform(vectorData).as[ResultKMeans]

        val distIntraClusters = for (cluster <- 0 to k - 1) yield {
          val centroide = modelKMeans.clusterCenters(cluster)
          val datosCluster = clusteredData.filter(_.prediction == cluster)
          val numDatosCluster = datosCluster.count()
          val distPuntosCentroide = datosCluster.map(x => Vectors.sqdist(centroide, x.features)).rdd.sum
          math.sqrt(distPuntosCentroide / numDatosCluster)
        }

        var sumElements = 0.0

        for (cluster1 <- 0 to k - 1) {
          val centroide1 = modelKMeans.clusterCenters(cluster1)
          val d1 = distIntraClusters(cluster1)
          var maxResult = 0.0
          for (cluster2 <- 0 to k - 1 if (cluster2 != cluster1)) {
            val centroide2 = modelKMeans.clusterCenters(cluster2)
            val d2 = distIntraClusters(cluster2)
            val d12 = math.sqrt(Vectors.sqdist(centroide1, centroide2))
            val result = (d1 + d2) / d12
            if (result > maxResult) maxResult = result
          }
          sumElements = sumElements + maxResult
        }

        val DBIndex = sumElements / k
        DBIndexesKMeans += Tuple2(k, DBIndex)
      }

      // BISECTING KMEANS
      if (modelBisectingKMeans != null) {
        val clusteredData = modelBisectingKMeans.transform(vectorData).as[ResultKMeans]

        val distIntraClusters = for (cluster <- 0 to k - 1) yield {
          val centroide = modelBisectingKMeans.clusterCenters(cluster)
          val datosCluster = clusteredData.filter(_.prediction == cluster)
          val numDatosCluster = datosCluster.count()
          val distPuntosCentroide = datosCluster.map(x => Vectors.sqdist(centroide, x.features)).rdd.sum
          math.sqrt(distPuntosCentroide / numDatosCluster)
        }

        var sumElements = 0.0

        for (cluster1 <- 0 to k - 1) {
          val centroide1 = modelBisectingKMeans.clusterCenters(cluster1)
          val d1 = distIntraClusters(cluster1)
          var maxResult = 0.0
          for (cluster2 <- 0 to k - 1 if (cluster2 != cluster1)) {
            val centroide2 = modelBisectingKMeans.clusterCenters(cluster2)
            val d2 = distIntraClusters(cluster2)
            val d12 = math.sqrt(Vectors.sqdist(centroide1, centroide2))
            val result = (d1 + d2) / d12
            if (result > maxResult) maxResult = result
          }
          sumElements = sumElements + maxResult
        }

        val DBIndex = sumElements / k
        DBIndexesBisectingKMeans += Tuple2(k, DBIndex)
      }

      // MEZCLAS GAUSSIANAS
      if (modelGMM != null) {
        val clusteredData = modelGMM.transform(vectorData).as[ResultGMM]

        var numClustersFinales = 0
        val distIntraClusters = for (cluster <- 0 to k - 1) yield {
          val centroide = modelGMM.gaussians(cluster).mean
          val datosCluster = clusteredData.filter(_.prediction == cluster)
          val numDatosCluster = datosCluster.count()
          if (numDatosCluster > 0) {
            numClustersFinales = numClustersFinales + 1
            val distPuntosCentroide = datosCluster.map(x => Vectors.sqdist(centroide, x.features)).rdd.sum
            math.sqrt(distPuntosCentroide / numDatosCluster)
          } else 0.0
        }

        var sumElements = 0.0

        for (cluster1 <- 0 to k - 1) {
          val centroide1 = modelGMM.gaussians(cluster1).mean
          val d1 = distIntraClusters(cluster1)
          var maxResult = 0.0
          if (d1 > 0.0) {
            for (cluster2 <- 0 to k - 1 if (cluster1 != cluster2)) {
              val centroide2 = modelGMM.gaussians(cluster2).mean
              val d2 = distIntraClusters(cluster2)
              if (d2 > 0.0) {
                val d12 = math.sqrt(Vectors.sqdist(centroide1, centroide2))
                val result = (d1 + d2) / d12
                if (result > maxResult) maxResult = result
              }
            }
            sumElements = sumElements + maxResult
          }
        }

        val DBIndex = sumElements / numClustersFinales
        DBIndexesGMM += Tuple2(numClustersFinales, DBIndex)
      }
    }

    val listResultFinal = ListBuffer.empty[ResultIndex]

    if (!DBIndexesKMeans.isEmpty) {
      val result = DBIndexesKMeans.sortBy(x => x._2).head
      listResultFinal += ResultIndex(ClusteringIndexes.METHOD_KMEANS, ClusteringIndexes.INDEX_DB, result._1, result._2)
    }

    if (!DBIndexesBisectingKMeans.isEmpty) {
      val result = DBIndexesBisectingKMeans.sortBy(x => x._2).head
      listResultFinal += ResultIndex(ClusteringIndexes.METHOD_BISECTING_KMEANS, ClusteringIndexes.INDEX_DB, result._1, result._2)
    }

    if (!DBIndexesGMM.isEmpty) {
      val result = DBIndexesGMM.sortBy(x => x._2).head
      listResultFinal += ResultIndex(ClusteringIndexes.METHOD_GMM, ClusteringIndexes.INDEX_DB, result._1, result._2)
    }

    listResultFinal
  }
}