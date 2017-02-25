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

    val dsExp1 = Spark.spark.read.parquet("TGAS-Exp1")
    val vectorData = new VectorAssembler().setInputCols(Array("ra", "dec", "pmra")).setOutputCol("features").transform(dsExp1).select("tycho2_id", "features").cache

    val numInstances = vectorData.count

    println(s"N = $numInstances")

    val maxIterations = 20
    val k = 500

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

    //    val n = 100
    //    val indexedDataset = Spark.spark.sqlContext.range(n).withColumn("feature1", rand).withColumn("feature2", rand).withColumn("feature3", rand).withColumn("feature4", rand).distinct.rdd
    //    indexedDataset.count
    //    val tIni4 = new Date().getTime
    //
    //    val distancesRdd = indexedDataset.cartesian(indexedDataset).flatMap { x =>
    //      val index1 = x._1.getAs[Long]("id")
    //      val index2 = x._2.getAs[Long]("id")
    //      if (index1 < index2) {
    //        val vectorValues1 = Vectors.dense(x._1.toSeq.tail.toArray.map(x => x.toString().toDouble))
    //        val vectorValues2 = Vectors.dense(x._2.toSeq.tail.toArray.map(x => x.toString().toDouble))
    //        val distance = sim(vectorValues1, vectorValues2)
    //        Some(index1, index2, distance)
    //      } else {
    //        None
    //      }
    //    }

    //    println("distancesRdd count: "+distancesRdd.count)
    //    val model = new PowerIterationClustering().setK(k).setMaxIterations(maxIterations).setInitializationMode("degree").run(distancesRdd)
    //    val clusters = model.assignments.collect().groupBy(_.cluster).mapValues(_.map(_.id)).toList
    //    clusters.toList.foreach(x => println(x._1 + ": " + x._2.size))
    //
    //    val tFin4 = new Date().getTime
    //
    //    val tEmpleado4 = (tFin4 - tIni4) / 1000.0

    //    println(f"El proceso ha finalizado con KMeans ha sido $tEmpleado%2.2f segundos")
    //    println(f"El proceso ha finalizado con BisectingKMeans ha sido $tEmpleado2%2.2f segundos")
    //    println(f"El proceso ha finalizado con GaussianMixgures ha sido $tEmpleado3%2.2f segundos")
    //    println(f"El proceso ha finalizado con PowerIterationClustering ha sido $tEmpleado4%2.2f segundos")

    Spark.spark.stop()
  }

  private def sim(x: org.apache.spark.ml.linalg.Vector, y: org.apache.spark.ml.linalg.Vector): Double = {
    val dist2 = math.pow(x(0) - y(0), 2) * math.pow(x(1) - y(1), 2) * math.pow(x(2) - y(2), 2) * math.pow(x(3) - y(3), 2)
    math.exp(-dist2 / 4.0)
  }
}