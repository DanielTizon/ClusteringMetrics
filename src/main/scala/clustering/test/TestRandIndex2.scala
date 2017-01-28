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
    val k = 20

    val res = new KMeans().setK(k).setMaxIter(maxIterations).fit(scaledDS).transform(scaledDS)
    //val res = new BisectingKMeans().setK(k).setMaxIter(maxIterations).fit(scaledDS).transform(scaledDS)
    //val res = new GaussianMixture().setK(k).setMaxIter(maxIterations).fit(scaledDS).transform(scaledDS)
    
    //res.select("id", "prediction").join(dataset, "id").sample(false, 0.40).coalesce(1).write.option("header", true).csv("/home/tornar/clustering_15_grupos")
    
    val evidencia = spark.read.option("header", true).csv("/home/tornar/Dropbox/Inteligencia Artificial/TFM/ValidacionExterna.csv").rdd.map(x => (x.getAs[String]("GRUPO"), x.getAs[String]("ID")))

    val evidenciaAgrupados: RDD[Tuple3[Long, String, String]] = evidencia.groupByKey.flatMap(x => x._2.toSet.subsets(2)).map(x => (x.head, x.last)).zipWithIndex().map(x => (x._2, x._1._1, x._1._2))
    val evidenciaSeparados: RDD[Tuple3[Long, String, String]] = evidencia.cartesian(evidencia).filter(x => x._1._1 != x._2._1).map(x => (x._1._2, x._2._2)).zipWithIndex().map(x => (x._2, x._1._1, x._1._2))

    val tIni = new Date().getTime
    val randIndex = IndexRand.calculate(res, evidenciaAgrupados, evidenciaSeparados)
    val tFin = new Date().getTime
    val tEmpleado = (tFin - tIni) / 1000.0
    println("Rand Index: " + randIndex)
    println("Tiempo empleado: " + tEmpleado)
  }

}