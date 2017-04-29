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
import org.apache.spark.sql.functions._
import org.apache.spark.sql.SQLContext

object TestGaia {

  def main(args: Array[String]) {

    val spark = Spark.spark
    
    import spark.implicits._

    Spark.conf.setAppName("Gaia-Clustering-Metrics")
      //.setMaster("local[*]")
      .set("spark.ui.port", "2001")
      .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
      .set("spark.sql.warehouse.dir", "spark-warehouse")

    val numRepeticiones = 1
    val maxIterations = 200
    val minProbabilityGMM = 0.75

    val indexes = Seq(ClusteringIndexes.INDEX_BALL, ClusteringIndexes.INDEX_CH, ClusteringIndexes.INDEX_DB, ClusteringIndexes.INDEX_HARTIGAN,
      ClusteringIndexes.INDEX_KL, ClusteringIndexes.INDEX_RATKOWSKY)

    val method = ClusteringIndexes.METHOD_GMM

    val seqK = 5 to 100 by 1

    val evidencia = Spark.spark.read.option("header", true).csv("validacion_externa_tgas.csv")

    val dsGrupo = standarize(spark.read.parquet("DS_GRUPO4").drop("prediction")).withColumnRenamed("tycho2", "ID")
    

    val dsExp = Spark.spark.read.parquet("TGAS-Exp2").withColumn("errorRelativeParallax", 'parallax_error / 'parallax).where("errorRelativeParallax < 0.20")
    val vectorData = new VectorAssembler().setInputCols(Array("X", "Y", "Z", "VTA", "VTD")).setOutputCol("features").transform(dsExp)
    val scaledVectorData = removeOutliers(standarize(vectorData), 5)

    val tIni1 = new Date().getTime
    val result1 = ClusteringIndexes.estimateNumberClusters(dsGrupo, seqK.toList, indexes = indexes, method = method, repeticiones = numRepeticiones,
      minProbabilityGMM = minProbabilityGMM, maxIterations = maxIterations, evidencia = evidencia)

    println(result1.sortBy(x => x.points).reverse.mkString("\n"))

    val tFin1 = new Date().getTime
    val tEmpleado1 = (tFin1 - tIni1) / (60000.0 * numRepeticiones)
    println(s"TIEMPO EMPLEADO EXPERIMENTO: $tEmpleado1")

    Spark.spark.stop()
  }
  
  /**
   * CALCULO DE COORDENADAS
   */
  def getX = udf { (parallax: Double, ascensionRecta: Double, declinacion: Double) =>
    val distance = 1 / parallax
    distance * scala.math.cos(ascensionRecta) * scala.math.cos(declinacion)
  }

  def getY = udf { (parallax: Double, ascensionRecta: Double, declinacion: Double) =>
    val distance = 1 / parallax
    distance * scala.math.sin(ascensionRecta) * scala.math.cos(declinacion)
  }

  def getZ = udf { (parallax: Double, ascensionRecta: Double, declinacion: Double) =>
    val distance = 1 / parallax
    distance * scala.math.sin(declinacion)
  }

  /**
   * CALCULO DE VELOCIDAD TRANSVERSAL
   */
  def getVelocidadTransversal = udf { (parallax: Double, movPropioMilis: Double) =>
    val distance = 1 / parallax
    val movPropioSecs = movPropioMilis / 1000
    movPropioSecs * distance * 4.74
  }

  /**
   * GENERACION DEL DATAFRAME DE TGAS PARA EXPERIMENTO 1
   */
  def generateTGASExpDF1(sqlContext: SQLContext) = {
    import Spark.spark.implicits._
    val rutaDatosOriginales = "TGAS-DR1-parquet"
    val rutaFinal = "TGAS-Exp1"
    val columns = Seq("hip", "tycho2_id", "solution_id", "source_id", "ra", "dec", "pmra", "pmdec", "parallax")
    val columnsDouble = Seq("ra", "dec", "pmra", "pmdec", "parallax")
    var tgasDF = sqlContext.read.parquet(rutaDatosOriginales).select(columns.map(col): _*)
    columnsDouble.foreach(x => tgasDF = tgasDF.withColumn(x, 'x.cast("Double")))
    tgasDF.write.parquet(rutaFinal)
  }

  /**
   * GENERACION DEL DATAFRAME DE TGAS PARA EXPERIMENTO 2
   */
  def generateTGASExpDF2(sqlContext: SQLContext) = {

    import Spark.spark.implicits._

    val rutaDatosOriginales = "TGAS-DR1-parquet"
    val rutaFinal = "TGAS-Exp2"
    val columns = Seq("hip", "tycho2_id", "solution_id", "source_id", "ra", "dec", "pmra", "pmdec", "parallax")
    val columnsDouble = Seq("ra", "dec", "pmra", "pmdec", "parallax")
    var tgasDF = sqlContext.read.parquet(rutaDatosOriginales).select(columns.map(col): _*)
    columnsDouble.foreach(x => tgasDF = tgasDF.withColumn(x, 'x.cast("Double")))
    tgasDF.withColumn("X", getX('parallax, 'ra, 'dec)).withColumn("Y", getY('parallax, 'ra, 'dec)).withColumn("Z", getZ('parallax, 'ra, 'dec))
      .withColumn("VTA", getVelocidadTransversal('parallax, 'pmra)).withColumn("VTD", getVelocidadTransversal('parallax, 'pmdec))
      .drop("ra", "dec", "parallax", "pmra", "pmdec")
      .write.parquet(rutaFinal)
  }
}