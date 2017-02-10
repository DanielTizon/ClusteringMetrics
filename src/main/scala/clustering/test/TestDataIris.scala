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

    Spark.conf.setAppName("Test Data Iris")
      .setMaster("local[*]")
      .set("spark.sql.warehouse.dir", "spark-warehouse")

    val rutaCSV = "iris.csv"

    val ds = Spark.spark.read.option("delimiter", ",").option("header", "true").csv(rutaCSV)
      .withColumn("SepalLength", col("SepalLength").cast("Double"))
      .withColumn("SepalWidth", col("SepalWidth").cast("Double"))
      .withColumn("PetalLength", col("PetalLength").cast("Double"))
      .withColumn("PetalWidth", col("PetalWidth").cast("Double"))
      .withColumnRenamed("index", "id")

    val vectorData = new VectorAssembler().setInputCols(Array("SepalLength", "SepalWidth", "PetalLength", "PetalWidth")).setOutputCol("features").transform(ds).select("id", "features")

    // EVALUACION INTERNA
    val numRepeticiones = 1
    val maxIterations = 20
    val method = ClusteringIndexes.METHOD_KMEANS
    val index = ClusteringIndexes.INDEX_BALL

    val tIni = new Date().getTime
    val result = ClusteringIndexes.estimateNumberClusters(vectorData, (2 to 15).toList, index = index, method = method, repeticiones = numRepeticiones)
    println(s"${result.sortBy(x => x.points).reverse.mkString("\n")}")

    val tFin = new Date().getTime
    val tEmpleado = (tFin - tIni) / 1000.0
    println(s"\nEl proceso ha finalizado en $tEmpleado segundos")

    // EVALUACION EXTERNA - RAND INDEX
    //    val evidencia = Spark.spark.read.option("header", true).csv("Validacion Externa Iris.csv")
    //    val res = new GaussianMixture().setK(3).fit(vectorData).transform(vectorData)
    //    res.show
    //    val randIndex = IndexRand.calculate(res, evidencia)
    //    println("Rand Index: " + randIndex)

    Spark.spark.stop()
  }
}