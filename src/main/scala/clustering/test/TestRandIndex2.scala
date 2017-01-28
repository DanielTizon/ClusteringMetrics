package clustering.test

import java.util.Date

import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.functions.col

import clustering.metrics.ClusteringIndexes
import clustering.metrics.Spark
import clustering.metrics.indexes.IndexRand
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.clustering.BisectingKMeans
import org.apache.spark.ml.clustering.GaussianMixture
import org.apache.spark.mllib.clustering.PowerIterationClustering
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.rdd.RDD
import clustering.metrics.Utils.standarize
import org.apache.spark.sql.functions.desc

object TestRandIndex2 {
  def main(args: Array[String]) {

    //    Spark.conf.setAppName("Clustering-metrics")
    //      .set("spark.ui.port", "1982")
    //      .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    //      .set("spark.sql.warehouse.dir", "/user/dtizon/spark-warehouse")

    Spark.conf.setAppName("Test Data Iris")
      .setMaster("local[*]")
      .set("spark.sql.warehouse.dir", "spark-warehouse")

    val spark = Spark.spark

    val dataset = spark.read.parquet("/home/tornar/TGAS-Exp1").withColumnRenamed("tycho2_id", "id")
    val vectorData = new VectorAssembler().setInputCols(Array("ra", "dec", "pmra", "pmdec", "parallax")).setOutputCol("features").transform(dataset).select("id", "features")

    // Estandarizar datos
    val scaledDS = standarize(vectorData).cache

    val maxIterations = 20
    val k = 15

    val res = new KMeans().setK(k).setMaxIter(maxIterations).fit(scaledDS).transform(scaledDS)
    //val res = new BisectingKMeans().setK(k).setMaxIter(maxIterations).fit(scaledDS).transform(scaledDS)
    //val res = new GaussianMixture().setK(k).setMaxIter(maxIterations).fit(scaledDS).transform(scaledDS)


    //res.select("id", "prediction").join(dataset, "id").sample(false, 0.30).coalesce(1).write.option("header", true).csv("/home/tornar/TGAS_GROUPED")

    val evidencia = spark.read.option("header", true).csv("/home/tornar/Dropbox/Inteligencia Artificial/TFM/ValidacionExterna.csv")

    val tIni = new Date().getTime
    val randIndex = IndexRand.calculate(res, evidencia)
    val tFin = new Date().getTime
    val tEmpleado = (tFin - tIni) / 1000.0

    println("Rand Index: " + randIndex)
    println("Tiempo empleado: " + tEmpleado)
  }

}