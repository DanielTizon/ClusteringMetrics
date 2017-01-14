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

object TestComplexity {

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
      .withColumnRenamed("index", "id")//.sample(true, 1000.0)

    val vectorData = new VectorAssembler().setInputCols(Array("SepalLength", "SepalWidth", "PetalLength", "PetalWidth")).setOutputCol("features")
      .transform(ds).select("id", "features").cache

    val numInstances = vectorData.count

    println(s"N = $numInstances")

    val maxIterations = 20
    val k = 3

//    val tIni = new Date().getTime
//    val res1 = new KMeans().setK(k).setMaxIter(maxIterations).fit(vectorData).transform(vectorData)
//    res1.count
//    val tFin = new Date().getTime
//
//    val tEmpleado = (tFin - tIni) / 1000.0
//
//    val tIni2 = new Date().getTime
//    val res2 = new BisectingKMeans().setK(k).setMaxIter(maxIterations).fit(vectorData).transform(vectorData)
//    res2.count
//    val tFin2 = new Date().getTime
//
//    val tEmpleado2 = (tFin2 - tIni2) / 1000.0

    val tIni3 = new Date().getTime
    val model = new GaussianMixture().setK(k).setMaxIter(maxIterations).fit(vectorData)
    model.gaussians.foreach(x => {
      println("MEDIA: "+x.mean)
      println("COVARIANZA: "+x.cov.toString())
      
    })
    val weights = model.weights
    println("WEIGHTS: "+weights.mkString(","))
    val res3 = model.transform(vectorData)    
    res3.count
    val tFin3 = new Date().getTime

    val tEmpleado3 = (tFin3 - tIni3) / 1000.0

    //    val indexedDataset = ds.drop("Specie", "id").rdd.zipWithIndex().cache
    //    indexedDataset.count
    //    val tIni4 = new Date().getTime
    //
    //    val distancesRdd = indexedDataset.cartesian(indexedDataset).flatMap { x =>
    //      val index1 = x._1._2
    //      val index2 = x._2._2
    //      if (index1 < index2) {
    //        val vectorValues1 = Vectors.dense(x._1._1.toSeq.toArray.map(x => x.toString().toDouble))
    //        val vectorValues2 = Vectors.dense(x._2._1.toSeq.toArray.map(x => x.toString().toDouble))
    //        val distance = sim(vectorValues1, vectorValues2)
    //        Some(index1, index2, distance)
    //      } else {
    //        None
    //      }
    //    }
    //
    //    val model = new PowerIterationClustering().setK(k).setMaxIterations(maxIterations).setInitializationMode("degree").run(distancesRdd)
    //    val clusters = model.assignments.collect().groupBy(_.cluster).mapValues(_.map(_.id)).toList
    //    clusters.toList.foreach(x => println(x._1 + ": " + x._2.size))
    //
    //    val tFin4 = new Date().getTime
    //
    //    val tEmpleado4 = (tFin4 - tIni4) / 1000.0

//    println(f"El proceso ha finalizado con KMeans ha sido $tEmpleado%2.2f segundos")
//    println(f"El proceso ha finalizado con BisectingKMeans ha sido $tEmpleado2%2.2f segundos")
    println(f"El proceso ha finalizado con GaussianMixgures ha sido $tEmpleado3%2.2f segundos")
    //    println(f"El proceso ha finalizado con PowerIterationClustering ha sido $tEmpleado4%2.2f segundos")

    Spark.spark.stop()
  }

  private def sim(x: org.apache.spark.ml.linalg.Vector, y: org.apache.spark.ml.linalg.Vector): Double = {
    val dist2 = math.pow(x(0) - y(0), 2) * math.pow(x(1) - y(1), 2) * math.pow(x(2) - y(2), 2) * math.pow(x(3) - y(3), 2)
    math.exp(-dist2 / 4.0)
  }
}