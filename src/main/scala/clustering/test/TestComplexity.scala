package clustering.test

import java.util.Date

import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.functions.col

import clustering.metrics.ClusteringIndexes
import clustering.metrics.Spark
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.clustering.BisectingKMeans
import org.apache.spark.ml.clustering.GaussianMixture
import org.apache.spark.mllib.clustering.PowerIterationClustering
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.functions.rand

object TestComplexity {

  def main(args: Array[String]) {

    Spark.conf.setAppName("Test Complexity Order")
      //.setMaster("local[*]")
      .set("spark.ui.port", "2001")
      .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
      .set("spark.sql.warehouse.dir", "spark-warehouse")

    val dsExp1 = Spark.spark.read.parquet("/archive/parquet/gaia/gdr1/gaiaSource")
    val vectorData = new VectorAssembler().setInputCols(Array("ra", "dec", "pmdec","pmra", "parallax")).setOutputCol("features").transform(dsExp1).select("source_id", "features").cache

    val numInstances = vectorData.count

    println(s"N = $numInstances")

    val maxIterations = 100
    val k = 50

    val tIni = new Date().getTime
    val res1 = new KMeans().setK(k).setMaxIter(maxIterations).fit(vectorData).transform(vectorData)
    res1.count
    val tFin = new Date().getTime

    val tEmpleado = (tFin - tIni) / 1000.0
    print(s"Tiempo KMeans: $tEmpleado segundos")

    val tIni2 = new Date().getTime
    val res2 = new BisectingKMeans().setK(k).setMaxIter(maxIterations).fit(vectorData).transform(vectorData)
    res2.count
    val tFin2 = new Date().getTime

    val tEmpleado2 = (tFin2 - tIni2) / 1000.0
    print(s"Tiempo BisectingKMeans: $tEmpleado2 segundos")

    val tIni3 = new Date().getTime
    val res3 = new GaussianMixture().setK(k).setMaxIter(maxIterations).fit(vectorData).transform(vectorData)
    res3.count
    val tFin3 = new Date().getTime
    val tEmpleado3 = (tFin3 - tIni3) / 1000.0
    print(s"Tiempo GaussianMixture: $tEmpleado3 segundos")

    Spark.spark.stop()
  }

  private def sim(x: org.apache.spark.ml.linalg.Vector, y: org.apache.spark.ml.linalg.Vector): Double = {
    val dist2 = math.pow(x(0) - y(0), 2) * math.pow(x(1) - y(1), 2) * math.pow(x(2) - y(2), 2) * math.pow(x(3) - y(3), 2)
    math.exp(-dist2 / 4.0)
  }
}