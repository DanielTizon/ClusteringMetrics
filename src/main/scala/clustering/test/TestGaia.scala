package clustering.test

import java.util.Date

import org.apache.spark.ml.feature.VectorAssembler

import clustering.metrics.ClusteringIndexes
import clustering.metrics.ClusteringIndexes.getMax
import clustering.metrics.Spark
import clustering.metrics.Utils.standarize
import clustering.metrics.Utils.removeOutliers
import org.apache.spark.ml.clustering.KMeans
import clustering.metrics.indexes.IndexRand
import org.apache.spark.ml.clustering.GaussianMixture
import org.apache.spark.sql.functions.col

object TestGaia {

  def main(args: Array[String]) {

    Spark.conf.setAppName("Gaia-Clustering-Metrics")
      //.setMaster("local[*]")
      .set("spark.ui.port", "2001")
      .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
      .set("spark.sql.warehouse.dir", "spark-warehouse")

    val numRepeticiones = 1
    val maxIterations = 20
    val minProbabilityGMM = 0.75

    val index = ClusteringIndexes.INDEX_ALL
    val method = ClusteringIndexes.METHOD_GMM

    val seqK = 5 to 100 by 1

    val dsExp = Spark.spark.read.parquet("TGAS-Exp2")
    val vectorData = new VectorAssembler().setInputCols(Array("X", "Y", "Z", "VTA", "VTD")).setOutputCol("features").transform(dsExp).select("tycho2_id", "features")
    val scaledVectorData = removeOutliers(standarize(vectorData), 5)

    val tIni1 = new Date().getTime
    val result1 = ClusteringIndexes.estimateNumberClusters(scaledVectorData, seqK.toList, index = index, method = method, repeticiones = numRepeticiones, minProbabilityGMM = minProbabilityGMM)
    println(s"${result1.sortBy(x => x.points).reverse.mkString("\n")}")
    
    val tFin1 = new Date().getTime
    val tEmpleado1 = (tFin1 - tIni1) / (60000.0 * numRepeticiones)    
    println(s"TIEMPO EMPLEADO EXPERIMENTO: $tEmpleado1")

    // EVALUACION EXTERNA - RAND INDEX
    //    val evidencia = Spark.spark.read.option("header", true).csv("/home/tornar/Dropbox/Inteligencia Artificial/TFM/Validacion Externa Gaia.csv")
    //    val res = new GaussianMixture().setK(15).fit(scaledVectorData1).transform(scaledVectorData1).withColumn("GroupMaxProb", getMax(col("probability")))
    //      .withColumnRenamed("tycho2_id", "ID")
    //      .where("GroupMaxProb >= " + minProbabilityGMM)
    //    val randIndex = IndexRand.calculate(res, evidencia)
    //    println("Rand Index: " + randIndex)

    Spark.spark.stop()
  }
}