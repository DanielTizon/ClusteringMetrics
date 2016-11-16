package clustering.test

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import java.util.Date
import clustering.metrics.ClusteringIndexes
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.SparkSession
import clustering.metrics.Spark
import clustering.metrics.Results._
import org.apache.spark.mllib.stat.MultivariateStatisticalSummary
import org.apache.spark.mllib.stat.Statistics
import org.apache.spark.ml.feature.StandardScaler
import org.apache.spark.sql.Dataset
import clustering.metrics.Results
import clustering.metrics.Utils._

object TestIndexes {

  def main(args: Array[String]) {

    Spark.conf.setAppName("Clustering-metrics")
      .set("spark.ui.port", "1982")      
      .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
      .set("spark.sql.warehouse.dir", "/user/dtizon/spark-warehouse")

    import Spark.spark.implicits._

    val dsExp1 = Spark.spark.read.parquet("/user/dtizon/TGAS-Exp1").as[Experimento1]
    val dsExp2 = Spark.spark.read.parquet("/user/dtizon/TGAS-Exp2").as[Experimento2]

    val vectorData1 = dsExp1.map(x => VectorData(x.tycho2_id, Vectors.dense(x.ra, x.dec, x.pmra, x.pmdec, x.parallax)))
    val vectorData2 = dsExp2.map(x => VectorData(x.tycho2_id, Vectors.dense(x.x, x.y, x.z, x.vta, x.vtd)))

    // Estandarizar datos
    val scaledVectorData1 = standarize(vectorData1)
    val scaledVectorData2 = standarize(vectorData2)

    val numRepeticiones = 3
    val maxIterations = 20

    val index = ClusteringIndexes.INDEX_ALL
    val method = ClusteringIndexes.METHOD_GMM

    val tIni1 = new Date().getTime
    val result1 = ClusteringIndexes.estimateNumberClusters(scaledVectorData1, List(10, 20, 30, 40, 50, 60, 70, 80, 90, 100), index = index, method = method, repeticiones = numRepeticiones)
    val tFin1 = new Date().getTime
    val tEmpleado1 = (tFin1 - tIni1) / (60000.0 * numRepeticiones)

    println(s"EXPERIMENTO 1 - TIEMPO EMPLEADO: $tEmpleado1 minutos")
    result1.foreach(x => println(s"EXPERIMENTO 1: $x"))

    val tIni2 = new Date().getTime
    val result2 = ClusteringIndexes.estimateNumberClusters(scaledVectorData2, List(10, 20, 30, 40, 50, 60, 70, 80, 90, 100), index = index, method = method, repeticiones = numRepeticiones)
    val tFin2 = new Date().getTime
    val tEmpleado2 = (tFin2 - tIni2) / (60000.0 * numRepeticiones)

    println(s"EXPERIMENTO 2 - TIEMPO EMPLEADO: $tEmpleado2 minutos")
    result2.foreach(x => println(s"EXPERIMENTO 2: $x"))

    Spark.spark.stop()
  }
}