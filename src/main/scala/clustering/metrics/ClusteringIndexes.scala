package clustering.metrics

import org.apache.spark.ml.clustering.GaussianMixtureModel
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.mllib.clustering.KMeansModel
import scala.util.Try
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.sql.functions._
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.clustering.GaussianMixture
import org.apache.spark.sql.DataFrame
import org.apache.spark.mllib.linalg.distributed.{ CoordinateMatrix, MatrixEntry }
import breeze.linalg.DenseVector
import scala.collection.mutable.ListBuffer
import org.apache.spark.storage.StorageLevel
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.Row
import clustering.test.TestIndexes
import clustering.metrics.indexes._
import org.apache.spark.ml.clustering.BisectingKMeans
import org.apache.spark.ml.clustering.GaussianMixture
import clustering.metrics.Results._

object ClusteringIndexes {

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

  def estimateNumberClusters(vectorData: Dataset[Results.VectorData], seqK: List[Int] = (2 to 15).toList, index: String = INDEX_ALL, method: String = METHOD_KMEANS,
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

