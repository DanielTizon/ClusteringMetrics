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

object TestRandIndex {
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

    val vectorData = new VectorAssembler().setInputCols(Array("SepalLength", "SepalWidth", "PetalLength", "PetalWidth")).setOutputCol("features")
      .transform(ds).select("id", "features").cache

    val numInstances = vectorData.count

    println(s"N = $numInstances")

    val maxIterations = 20
    val k = 3

    val resKMeans = new KMeans().setK(k).setMaxIter(maxIterations).fit(vectorData).transform(vectorData)
    val evidenciaSetosa = Spark.spark.sparkContext.parallelize(1 to 50).cartesian(Spark.spark.sparkContext.parallelize(1 to 50)).filter(x => x._1 < x._2).zipWithIndex().map(x => (x._2, x._1._1.toString, x._1._2.toString))
    val evidenciaAgrupados: RDD[Tuple3[Long, String, String]] = evidenciaSetosa
    val evidenciaSeparados = Spark.spark.sparkContext.emptyRDD[Tuple3[Long, String, String]]
    val tIni = new Date().getTime
    val randIndex = IndexRand.calculate(resKMeans, evidenciaAgrupados, evidenciaSeparados)
    val tFin = new Date().getTime
    val tEmpleado = (tFin - tIni) / 1000.0
    println("Rand Index: " + randIndex)
    println("Tiempo empleado: " + tEmpleado)
  }

}