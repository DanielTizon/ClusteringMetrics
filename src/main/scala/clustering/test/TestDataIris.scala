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
      .withColumnRenamed("index", "ID")

    val vectorData = new VectorAssembler().setInputCols(Array("SepalLength", "SepalWidth", "PetalLength", "PetalWidth")).setOutputCol("features").transform(ds).select("ID", "features")

    val numRepeticiones = 1
    val maxIterations = 20
    val method = ClusteringIndexes.METHOD_KMEANS
    //val indexes = Seq(ClusteringIndexes.INDEX_CH, ClusteringIndexes.INDEX_DB, ClusteringIndexes.INDEX_HARTIGAN, ClusteringIndexes.INDEX_KL, ClusteringIndexes.INDEX_RATKOWSKY, ClusteringIndexes.INDEX_RAND)
    val indexes = Seq(ClusteringIndexes.INDEX_RAND)
    
    val evidencia = ds.select("Specie", "ID").withColumnRenamed("Specie", "GRUPO")

    val tIni = new Date().getTime
    val result = ClusteringIndexes.estimateNumberClusters(vectorData, (2 to 15).toList, indexes = indexes, method = method, repeticiones = numRepeticiones, evidencia = evidencia)
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