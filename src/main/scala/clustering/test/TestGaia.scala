package clustering.test

import java.util.Date

import org.apache.spark.ml.feature.VectorAssembler

import clustering.metrics.ClusteringIndexes
import clustering.metrics.Spark
import clustering.metrics.Utils.standarize
import org.apache.spark.ml.clustering.KMeans
import clustering.metrics.indexes.IndexRand
import org.apache.spark.ml.clustering.GaussianMixture

object TestIndexes {

  def main(args: Array[String]) {

    Spark.conf.setAppName("Gaia-Clustering-Metrics")
      //.setMaster("local[*]")
      .set("spark.ui.port", "2001")
      .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
      .set("spark.sql.warehouse.dir", "spark-warehouse")

    val dsExp1 = Spark.spark.read.parquet("TGAS-Exp1")
    val dsExp2 = Spark.spark.read.parquet("TGAS-Exp2")

    val vectorData1 = new VectorAssembler().setInputCols(Array("ra", "dec", "pmra", "pmdec", "parallax")).setOutputCol("features").transform(dsExp1).select("tycho2_id", "features")
    val vectorData2 = new VectorAssembler().setInputCols(Array("X", "Y", "Z", "VTA", "VTD")).setOutputCol("features").transform(dsExp2).select("tycho2_id", "features")

    // Estandarizar datos
    val scaledVectorData1 = standarize(vectorData1)
    val scaledVectorData2 = standarize(vectorData2)

    val numRepeticiones = 1
    val maxIterations = 20

    val index = ClusteringIndexes.INDEX_ALL
    val method = ClusteringIndexes.METHOD_KMEANS

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

    // EVALUACION EXTERNA - RAND INDEX
    val evidencia = Spark.spark.read.option("header", true).csv("/home/tornar/Dropbox/Inteligencia Artificial/TFM/Validacion Externa Gaia.csv")
    val res = new GaussianMixture().setK(20).fit(scaledVectorData1).transform(scaledVectorData1)
    val randIndex = IndexRand.calculate(res, evidencia)
    println("Rand Index: " + randIndex)

    Spark.spark.stop()
  }
}