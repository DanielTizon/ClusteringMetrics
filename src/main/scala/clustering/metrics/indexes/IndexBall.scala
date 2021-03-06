package clustering.metrics.indexes

import scala.collection.mutable.ListBuffer

import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.rdd.RDD.doubleRDDToDoubleRDDFunctions
import org.apache.spark.sql.DataFrame

import clustering.metrics.Spark
import clustering.metrics.ClusteringIndexes
import clustering.metrics.ClusteringIndexes.ResultIndex
import clustering.metrics.ClusteringIndexes.TuplaModelos

object IndexBall {
  /**
   * ********************************************************************************
   *                   			                                                        *
   * 		BALL INDEX (Ball and Hall) - 1965 	                 					 						  *
   *    																																						*
   *    Index = Wq / q																															*
   *    																		                                        *
   *    The largest difference between levels indicate the best solution.    	      *
   *  								                                 															*
   * ********************************************************************************
   */
  def calculate(modelTuples: List[TuplaModelos], vectorData: DataFrame) = {
    println(s"BALL INDEX -> ${modelTuples.map(_.k)}")

    import Spark.spark.implicits._

    val ballIndexesKMeans: ListBuffer[Tuple2[Int, Double]] = ListBuffer[Tuple2[Int, Double]]()
    val ballIndexesBisectingKMeans: ListBuffer[Tuple2[Int, Double]] = ListBuffer[Tuple2[Int, Double]]()
    val ballIndexesGMM: ListBuffer[Tuple3[Int, Double, Int]] = ListBuffer[Tuple3[Int, Double, Int]]()

    for (modelsK <- modelTuples) {
      val k = modelsK.k
      println(s"CALCULANDO BALL INDEX PARA k = $k")
      val modelKMeans = modelsK.modelKMeans
      val modelBisectingKMeans = modelsK.modelBisectingKMeans
      val modelGMM = modelsK.modelGMM

      // KMEANS
      if (modelKMeans != null) {
        // Wq = WSSSE (Within Set Sum of Squared Errors)
        val Wq = modelKMeans._1.computeCost(vectorData)
        val ballIndex = Wq / k
        ballIndexesKMeans += Tuple2(k, ballIndex)
      }

      // BISECTING KMEANS
      if (modelBisectingKMeans != null) {
        // Wq = WSSSE (Within Set Sum of Squared Errors)
        val Wq = modelBisectingKMeans._1.computeCost(vectorData)
        val ballIndex = Wq / k
        ballIndexesBisectingKMeans += Tuple2(k, ballIndex)
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

        val ballIndex = Wq / numClustersFinales
        ballIndexesGMM += Tuple3(numClustersFinales, ballIndex, k)
      }
    }

    val listResultFinal = ListBuffer.empty[ResultIndex]

    if (!ballIndexesKMeans.isEmpty) {
      val listaSlopes = ballIndexesKMeans.sortBy(x => x._1).sliding(2).map(x => {
        val ballIndex1 = x(0)
        val ballIndex2 = x(1)
        (ballIndex2._1, math.abs(ballIndex1._2 - ballIndex2._2))
      }).toList

      val result = listaSlopes.sortBy(x => x._2)
      var points = 0
      for (result_value <- result) {
        listResultFinal += ResultIndex(ClusteringIndexes.METHOD_KMEANS, ClusteringIndexes.INDEX_BALL, result_value._2, points, result_value._1, result_value._1)
        points = points + 1
      }
    }

    if (!ballIndexesBisectingKMeans.isEmpty) {
      val listaSlopes = ballIndexesBisectingKMeans.sortBy(x => x._1).sliding(2).map(x => {
        val ballIndex1 = x(0)
        val ballIndex2 = x(1)
        (ballIndex2._1, math.abs(ballIndex1._2 - ballIndex2._2))
      }).toList

      val result = listaSlopes.sortBy(x => x._2)
      var points = 0
      for (result_value <- result) {
        listResultFinal += ResultIndex(ClusteringIndexes.METHOD_BISECTING_KMEANS, ClusteringIndexes.INDEX_BALL, result_value._2, points, result_value._1, result_value._1)
        points = points + 1
      }
    }

    if (!ballIndexesGMM.isEmpty) {
      val listaSlopes = ballIndexesGMM.sortBy(x => x._1).sliding(2).map(x => {
        val ballIndex1 = x(0)
        val ballIndex2 = x(1)
        (ballIndex2._1, math.abs(ballIndex1._2 - ballIndex2._2), ballIndex2._3)
      }).toList

      val result = listaSlopes.sortBy(x => x._2)
      var points = 0
      for (result_value <- result) {
        listResultFinal += ResultIndex(ClusteringIndexes.METHOD_GMM, ClusteringIndexes.INDEX_BALL, result_value._2, points, result_value._3, result_value._1)
        points = points + 1
      }
    }

    listResultFinal
  }
}