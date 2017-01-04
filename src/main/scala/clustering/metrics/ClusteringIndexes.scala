package clustering.metrics

import scala.collection.mutable.ListBuffer

import org.apache.spark.ml.clustering.BisectingKMeans
import org.apache.spark.ml.clustering.GaussianMixture
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.DataFrame

import clustering.metrics.indexes.IndexBall
import clustering.metrics.indexes.IndexCH
import clustering.metrics.indexes.IndexDB
import clustering.metrics.indexes.IndexHartigan
import clustering.metrics.indexes.IndexKL
import clustering.metrics.indexes.IndexRatkowsky
import org.apache.spark.ml.clustering.KMeansModel
import org.apache.spark.ml.clustering.BisectingKMeansModel
import org.apache.spark.ml.clustering.GaussianMixtureModel

object ClusteringIndexes {
  
  case class TuplaModelos(k: Int, modelKMeans: KMeansModel, modelBisectingKMeans: BisectingKMeansModel, modelGMM: GaussianMixtureModel)
  case class ResultIndex(val method: String, val indexType: String, val winnerK: Int, val indexValue: Double)

  val INDEX_BALL = "indexBall"
  val INDEX_CH = "indexCH"
  val INDEX_DB = "indexDB"
  val INDEX_HARTIGAN = "indexHartigan"
  val INDEX_RATKOWSKY = "indexRatkowsky"
  val INDEX_KL = "indexKL"
  val INDEX_ALL = "all"

  val METHOD_KMEANS = "kmeans"
  val METHOD_BISECTING_KMEANS = "bisecting_kmeans"
  val METHOD_GMM = "gmm"
  val METHOD_ALL = "all"

  def estimateNumberClusters(vectorData: DataFrame, seqK: List[Int] = (2 to 15).toList, index: String = INDEX_ALL, method: String = METHOD_KMEANS,
                             repeticiones: Int = 1, maxIterations: Int = 20): List[ResultIndex] = {

    vectorData.cache()

    val resultadoFinal = ListBuffer[ResultIndex]()

    for (rep <- 1 to repeticiones) {

      val tupleModels = for (k <- seqK) yield {
        val modelKMeans = if (method != null && (method == METHOD_KMEANS || method == METHOD_ALL)) {
          new KMeans().setK(k).setMaxIter(maxIterations).fit(vectorData)
        } else null

        val modelBisectingKMeans = if (method != null && (method == METHOD_BISECTING_KMEANS || method == METHOD_ALL)) {
          new BisectingKMeans().setK(k).setMaxIter(maxIterations).fit(vectorData)
        } else null

        val modelGMM = if (method != null && (method == METHOD_GMM || method == METHOD_ALL)) {
          new GaussianMixture().setK(k).setMaxIter(maxIterations).fit(vectorData)
        } else null

        TuplaModelos(k, modelKMeans, modelBisectingKMeans, modelGMM)
      }

      if (index != null && (index == INDEX_BALL || index == INDEX_ALL)) {
        resultadoFinal ++= IndexBall.calculate(tupleModels, vectorData)
      }

      if (index != null && (index == INDEX_CH || index == INDEX_ALL)) {
        resultadoFinal ++= IndexCH.calculate(tupleModels, vectorData)
      }

      if (index != null && (index == INDEX_DB || index == INDEX_ALL)) {
        resultadoFinal ++= IndexDB.calculate(tupleModels, vectorData)
      }

      if (index != null && (index == INDEX_HARTIGAN || index == INDEX_ALL)) {
        resultadoFinal ++= IndexHartigan.calculate(tupleModels, vectorData)
      }

      if (index != null && (index == INDEX_RATKOWSKY || index == INDEX_ALL)) {
        resultadoFinal ++= IndexRatkowsky.calculate(tupleModels, vectorData)
      }

      if (index != null && (index == INDEX_KL || index == INDEX_ALL)) {
        val newSeqK = ((seqK.sortBy(x => x).head - 1) :: seqK) :+ (seqK.sortBy(x => x).last + 1)

        val newTupleModels = for (k <- List(newSeqK.head, newSeqK.last)) yield {
          val modelKMeans = if (k > 1 && method != null && (method == METHOD_KMEANS || method == METHOD_ALL)) {
            new KMeans().setK(k).setMaxIter(maxIterations).fit(vectorData)
          } else null

          val modelBisectingKMeans = if (k > 1 && method != null && (method == METHOD_BISECTING_KMEANS || method == METHOD_ALL)) {
            new BisectingKMeans().setK(k).setMaxIter(maxIterations).fit(vectorData)
          } else null

          val modelGMM = if (k > 1 && method != null && (method == METHOD_GMM || method == METHOD_ALL)) {
            new GaussianMixture().setK(k).setMaxIter(maxIterations).fit(vectorData)
          } else null

          TuplaModelos(k, modelKMeans, modelBisectingKMeans, modelGMM)
        }

        resultadoFinal ++= IndexKL.calculate(tupleModels ::: newTupleModels, vectorData)
      }
    }
    resultadoFinal.toList
  }
}

