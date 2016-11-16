package clustering.metrics

import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.clustering.KMeansModel
import org.apache.spark.ml.clustering.BisectingKMeansModel
import org.apache.spark.ml.clustering.GaussianMixtureModel

object Results {
  case class VectorData(id: String, features: Vector)
  case class ResultGMM(id: String, features: Vector, prediction: Int, probability: Vector)
  case class ResultKMeans(id: String, features: Vector, prediction: Int)
  case class ResultIndex(val method: String, val indexType: String, val winnerK: Int, val indexValue: Double)
  case class TuplaModelos(k: Int, modelKMeans: KMeansModel, modelBisectingKMeans: BisectingKMeansModel, modelGMM: GaussianMixtureModel)
}