package clustering.metrics.indexes

import scala.collection.mutable.ListBuffer

import org.apache.spark.rdd.RDD.doubleRDDToDoubleRDDFunctions
import org.apache.spark.sql.DataFrame

import clustering.metrics.Spark
import clustering.metrics.ClusteringIndexes
import clustering.metrics.ClusteringIndexes.ResultIndex
import clustering.metrics.ClusteringIndexes.TuplaModelos

object IndexRatkowsky {
  /**
   * ***************************************************************************
   *                   			                                                   *
   * 	  RATKOWSKY INDEX (Ratkowsky and Lance) - 1978			                     *
   *    																																			 *
   *    Index = S / sqrt(q)																		 								 *
   *    																																			 *
   *    S = sqrt(1/p * (SUMp BGSSp/TSSp))                                      *
   *    																																			 *
   *    BGSSp = SUMq nq * pow((Cqp - Mean-p), 2)															 *
   *    																																			 *
   *    TSSp = SUMn pow((Xnp - Mean-p), 2)																		 *
   *    																																			 *
   *    The maximum value of the index indicates the best solution             *
   *    																																			 *
   * ***************************************************************************
   */
  def calculate(modelTuples: List[TuplaModelos], vectorData: DataFrame) = {
    println(s"RATKOWSKY INDEX -> ${modelTuples.map(_.k)}")


    val numVariables =  vectorData.head().getAs[org.apache.spark.ml.linalg.Vector]("features").size

    val ratkowskyIndexesKMeans: ListBuffer[Tuple2[Int, Double]] = ListBuffer[Tuple2[Int, Double]]()
    val ratkowskyIndexesBisectingKMeans: ListBuffer[Tuple2[Int, Double]] = ListBuffer[Tuple2[Int, Double]]()
    val ratkowskyIndexesGMM: ListBuffer[Tuple2[Int, Double]] = ListBuffer[Tuple2[Int, Double]]()

    for (modelsK <- modelTuples) {
      val k = modelsK.k
      val modelKMeans = modelsK.modelKMeans
      val modelBisectingKMeans = modelsK.modelBisectingKMeans
      val modelGMM = modelsK.modelGMM

      // KMEANS
      if (modelKMeans != null) {
        val clusteredData = modelKMeans.transform(vectorData)
        val allNk = for (cluster <- 0 to k - 1) yield {
          clusteredData.where("prediction = "+cluster).count()
        }

        var result = 0.0
        for (variable <- 0 to numVariables - 1) {
          val varData = vectorData.rdd.map(x => x.getAs[org.apache.spark.ml.linalg.Vector]("features")(variable))
          val mediaVariable = varData.mean()
          var BGSSvar = 0.0
          for (cluster <- 0 to k - 1) {
            val ckj = modelKMeans.clusterCenters(cluster)(variable)
            val nk = allNk(cluster)
            BGSSvar = BGSSvar + nk * math.pow(ckj - mediaVariable, 2)
          }

          val TSSvar = varData.map(x => math.pow(x - mediaVariable, 2)).sum

          result = result + (BGSSvar / TSSvar)

        }

        val ratkowskyIndex = math.sqrt((result / numVariables) / k)
        ratkowskyIndexesKMeans += Tuple2(k, ratkowskyIndex)
      }

      // BISECTING KMEANS
      if (modelBisectingKMeans != null) {
        val clusteredData = modelBisectingKMeans.transform(vectorData)
        val allNk = for (cluster <- 0 to k - 1) yield {
          clusteredData.where("prediction = "+cluster).count()
        }

        var result = 0.0
        for (variable <- 0 to numVariables - 1) {
          val varData = vectorData.rdd.map(x => x.getAs[org.apache.spark.ml.linalg.Vector]("features")(variable))
          val mediaVariable = varData.mean()
          var BGSSvar = 0.0
          for (cluster <- 0 to k - 1) {
            val ckj = modelBisectingKMeans.clusterCenters(cluster)(variable)
            val nk = allNk(cluster)
            BGSSvar = BGSSvar + nk * math.pow(ckj - mediaVariable, 2)
          }

          val TSSvar = varData.map(x => math.pow(x - mediaVariable, 2)).sum

          result = result + (BGSSvar / TSSvar)

        }

        val ratkowskyIndex = math.sqrt((result / numVariables) / k)
        ratkowskyIndexesBisectingKMeans += Tuple2(k, ratkowskyIndex)
      }

      // MEZCLAS GAUSSIANAS
      if (modelGMM != null) {
        val clusteredData = modelGMM.transform(vectorData)

        var numClustersFinales = 0

        val allNk = for (cluster <- 0 to k - 1) yield {
          val numElementscluster = clusteredData.where("prediction ="+ cluster).count()
          if (numElementscluster > 0) { numClustersFinales = numClustersFinales + 1 }
          numElementscluster
        }

        var result = 0.0
        for (variable <- 0 to numVariables - 1) {
          val varData = vectorData.rdd.map(x => x.getAs[org.apache.spark.ml.linalg.Vector]("features")(variable))
          val mediaVariable = varData.mean()
          var BGSSvar = 0.0
          for (cluster <- 0 to k - 1) {
            val ckj = modelGMM.gaussians(cluster).mean(variable)
            val Nk = allNk(cluster)
            if (Nk > 0) {
              BGSSvar = BGSSvar + Nk * math.pow(ckj - mediaVariable, 2)
            }
          }

          val TSSvar = varData.map(x => math.pow(x - mediaVariable, 2)).sum
          result = result + (BGSSvar / TSSvar)
        }

        val ratkowskyIndex = math.sqrt((result / numVariables) / numClustersFinales)
        ratkowskyIndexesGMM += Tuple2(numClustersFinales, ratkowskyIndex)
      }
    }
    
    val listResultFinal = ListBuffer.empty[ResultIndex]

     if (!ratkowskyIndexesKMeans.isEmpty) {
      val result = ratkowskyIndexesKMeans.sortBy(x => x._2).last
      listResultFinal += ResultIndex(ClusteringIndexes.METHOD_KMEANS, ClusteringIndexes.INDEX_RATKOWSKY, result._1, result._2)
    }

    if (!ratkowskyIndexesBisectingKMeans.isEmpty) {
      val result = ratkowskyIndexesBisectingKMeans.sortBy(x => x._2).last
      listResultFinal += ResultIndex(ClusteringIndexes.METHOD_BISECTING_KMEANS, ClusteringIndexes.INDEX_RATKOWSKY, result._1, result._2)
    }
    
     if (!ratkowskyIndexesGMM.isEmpty) {
      val result = ratkowskyIndexesGMM.sortBy(x => x._2).last
      listResultFinal += ResultIndex(ClusteringIndexes.METHOD_GMM, ClusteringIndexes.INDEX_RATKOWSKY, result._1, result._2)
    }
     
     listResultFinal
  }
}