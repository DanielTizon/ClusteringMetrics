package clustering.test

import java.util.Date

import org.apache.spark.ml.feature.VectorAssembler

import clustering.metrics.ClusteringIndexes
import clustering.metrics.ClusteringIndexes.getMax
import clustering.metrics.Spark
import clustering.metrics.Utils.standarize
import clustering.metrics.Utils.removeOutliers
import clustering.metrics.Utils.getRelativeError
import org.apache.spark.ml.clustering.KMeans
import clustering.metrics.indexes.IndexRand
import org.apache.spark.ml.clustering.GaussianMixture
import org.apache.spark.sql.functions.col

object TestGaia {

  def main(args: Array[String]) {

    val spark = Spark.spark

    Spark.conf.setAppName("Gaia-Clustering-Metrics")
      //.setMaster("local[*]")
      .set("spark.ui.port", "2001")
      .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
      .set("spark.sql.warehouse.dir", "spark-warehouse")

    val numRepeticiones = 1
    val maxIterations = 200
    val minProbabilityGMM = 0.75

    val indexes = Seq(ClusteringIndexes.INDEX_BALL, ClusteringIndexes.INDEX_CH, ClusteringIndexes.INDEX_DB, ClusteringIndexes.INDEX_HARTIGAN,
      ClusteringIndexes.INDEX_KL, ClusteringIndexes.INDEX_RATKOWSKY, ClusteringIndexes.INDEX_RAND)

    val method = ClusteringIndexes.METHOD_GMM

    val seqK = 5 to 100 by 1

    val evidencia = Spark.spark.read.option("header", true).csv("validacion_externa_tgas.csv")
    
    val dsGrupo = standarize(spark.read.parquet("DS_GRUPO5").drop("prediction")).withColumnRenamed("tycho2", "ID")

    //    val dsExp = Spark.spark.read.parquet("TGAS-Exp2").withColumn("errorRelativeParallax", getRelativeError(col("parallax"), col("parallax_error"))).where("errorRelativeParallax < 0.20")
    //    val vectorData = new VectorAssembler().setInputCols(Array("X", "Y", "Z", "VTA", "VTD")).setOutputCol("features").transform(dsExp)
    //    val scaledVectorData = removeOutliers(standarize(vectorData), 5)

    val tIni1 = new Date().getTime
    val result1 = ClusteringIndexes.estimateNumberClusters(dsGrupo, seqK.toList, indexes = indexes, method = method, repeticiones = numRepeticiones,
      minProbabilityGMM = minProbabilityGMM, maxIterations = maxIterations, evidencia = evidencia)

    println(result1.sortBy(x => x.points).reverse.mkString("\n"))

    val tFin1 = new Date().getTime
    val tEmpleado1 = (tFin1 - tIni1) / (60000.0 * numRepeticiones)
    println(s"TIEMPO EMPLEADO EXPERIMENTO: $tEmpleado1")

    Spark.spark.stop()
  }
}