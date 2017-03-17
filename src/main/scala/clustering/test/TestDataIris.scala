package clustering.test

import java.util.Date

import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.functions.col

import clustering.metrics.ClusteringIndexes
import clustering.metrics.Spark
import clustering.metrics.indexes.IndexRand
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.clustering.GaussianMixture

object TestDataIris {

  def main(args: Array[String]) {

    import Spark.spark.implicits._

    Spark.conf.setAppName("Test Data Iris")
      .setMaster("local[*]")
      .set("spark.sql.warehouse.dir", "spark-warehouse")

    val numRepeticiones = 1
    val maxIterations = 20
    val method = ClusteringIndexes.METHOD_GMM
    val indexes = Seq(ClusteringIndexes.INDEX_RAND)

    val seqK = 2 to 15 by 1

    val ds = Spark.spark.read.option("delimiter", ",").option("header", "true").csv("iris.csv").withColumn("SepalLength", 'SepalLength.cast("Double"))
      .withColumn("SepalWidth", 'SepalWidth.cast("Double")).withColumn("PetalLength", 'PetalLength.cast("Double")).withColumn("PetalWidth", 'PetalWidth.cast("Double"))
      .withColumnRenamed("index", "ID")

    val vectorData = new VectorAssembler().setInputCols(Array("SepalLength", "SepalWidth", "PetalLength", "PetalWidth")).setOutputCol("features").transform(ds).select("ID", "features")

    val evidencia = Spark.spark.read.option("header", true).csv("validacion_externa_iris.csv")

    val tIni = new Date().getTime
    val result = ClusteringIndexes.estimateNumberClusters(vectorData, seqK.toList, indexes = indexes, method = method, repeticiones = numRepeticiones, evidencia = evidencia)
    println(s"${result.sortBy(x => x.points).reverse.mkString("\n")}")

    val tFin = new Date().getTime
    val tEmpleado = (tFin - tIni) / 1000.0
    println(s"\nEl proceso ha finalizado en $tEmpleado segundos")

    Spark.spark.stop()
  }
}