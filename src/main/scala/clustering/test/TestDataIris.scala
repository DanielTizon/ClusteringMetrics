package clustering.test

import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession
import clustering.metrics.ClusteringIndexes
import org.apache.spark.sql.types.StructType
import java.util.Date
import org.apache.spark.sql.types.StructField
import org.apache.spark.sql.Row
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.StringType
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.clustering.GaussianMixture
import clustering.metrics.Results._
import clustering.metrics.Spark
import org.apache.spark.ml.feature.Normalizer
import org.apache.spark.mllib.stat.MultivariateStatisticalSummary
import org.apache.spark.mllib.stat.Statistics
import org.apache.spark.ml.feature.StandardScaler
import clustering.metrics.Utils

object TestDataIris {

  case class Iris(index: String, SepalLength: Double, SepalWidth: Double, PetalLength: Double, PetalWidth: Double, Specie: String)

  def main(args: Array[String]) {

    Spark.conf.setAppName("Test Data Iris")
      .setMaster("local[*]")
      .set("spark.sql.warehouse.dir", "spark-warehouse")

    val rutaCSV = "iris.csv"

    import Spark.spark.implicits._

    val ds = Spark.spark.read.option("delimiter", ",").option("header", "true").csv(rutaCSV)
      .withColumn("SepalLength", col("SepalLength").cast("Double"))
      .withColumn("SepalWidth", col("SepalWidth").cast("Double"))
      .withColumn("PetalLength", col("PetalLength").cast("Double"))
      .withColumn("PetalWidth", col("PetalWidth").cast("Double"))
      .as[Iris]

    val vectorData = ds.map { x => VectorData(x.index, Vectors.dense(x.SepalLength, x.SepalWidth, x.PetalLength, x.PetalWidth)) }

    val numRepeticiones = 1
    val maxIterations = 20
    val method = ClusteringIndexes.METHOD_ALL
    val index = ClusteringIndexes.INDEX_ALL

    val tIni = new Date().getTime
    val result = ClusteringIndexes.estimateNumberClusters(vectorData, (2 to 15).toList, index = index, method = method, repeticiones = numRepeticiones)
    println(s"RESULT: $result")

    val resultFinal = result.groupBy(x => x.winnerK).map(x => (x._1, x._2.size)).toList.sortBy(x => x._2).reverse
    resultFinal.foreach(println)
    println(s"\nMAYORIA: ${resultFinal.head._1}")

    val tFin = new Date().getTime
    val tEmpleado = (tFin - tIni) / 1000.0
    println(s"El proceso ha finalizado en $tEmpleado segundos")

    Spark.spark.stop()
  }
}