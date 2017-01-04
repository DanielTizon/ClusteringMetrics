package clustering.test

import java.util.Date

import org.apache.spark.ml.feature.VectorAssembler

import clustering.metrics.ClusteringIndexes
import clustering.metrics.Spark
import clustering.metrics.Utils.standarize

object TestIndexes {

  def main(args: Array[String]) {

    Spark.conf.setAppName("Clustering-metrics")
      .set("spark.ui.port", "1982")      
      .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
      .set("spark.sql.warehouse.dir", "/user/dtizon/spark-warehouse")


    val dsExp1 = Spark.spark.read.parquet("/user/dtizon/TGAS-Exp1")
    val dsExp2 = Spark.spark.read.parquet("/user/dtizon/TGAS-Exp2")

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

    Spark.spark.stop()
  }
}