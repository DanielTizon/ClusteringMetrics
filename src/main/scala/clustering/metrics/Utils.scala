package clustering.metrics

import org.apache.spark.sql.functions._
import org.apache.spark.sql.SQLContext
import scala.reflect.runtime.universe
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.DataFrame
import org.apache.spark.mllib.stat.MultivariateStatisticalSummary
import org.apache.spark.ml.feature.StandardScaler
import org.apache.spark.mllib.stat.Statistics
import clustering.metrics.Results.VectorData

object Utils {

  case class Experimento1(hip: String, tycho2_id: String, ra: Double, dec: Double, pmra: Double, pmdec: Double, parallax: Double)
  case class Experimento2(hip: String, tycho2_id: String, x: Double, y: Double, z: Double, vta: Double, vtd: Double)

  def standarize(vectorData: Dataset[VectorData]): Dataset[VectorData] = {
    import Spark.spark.implicits._
    val scaler = new StandardScaler().setInputCol("features").setOutputCol("scaledFeatures").setWithStd(true).setWithMean(true)
    val scalerModel = scaler.fit(vectorData)
    scalerModel.transform(vectorData).drop("features").withColumnRenamed("scaledFeatures", "features").as[VectorData]
  }

  def summaryFeatures(vectorData: Dataset[VectorData]) = {
    val summary: MultivariateStatisticalSummary = Statistics.colStats(vectorData.rdd.map { x => org.apache.spark.mllib.linalg.Vectors.dense(x.features.toArray) })
    summary.min.toArray.foreach(x => println(f"MIN: $x%1.2f"))
    summary.mean.toArray.foreach(x => println(f"MEAN: $x%1.2f"))
    summary.variance.toArray.foreach(x => println(f"VARIANCE: $x%1.2f"))
    summary.max.toArray.foreach(x => println(f"MAX: $x%1.2f"))
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
    (movPropioSecs / 1000) * distance * 4.74
  }

  /**
   * GENERACION DEL DATAFRAME DE TGAS PARA EXPERIMENTO 1
   */
  def generateTGASExpDF1(sqlContext: SQLContext) = {
    val rutaDatosOriginales = "/user/dtizon/TGAS-DR1-parquet"
    val rutaFinal = "/user/dtizon/TGAS-Exp1"

    val columns = Seq("hip", "tycho2_id", "solution_id", "source_id", "ra", "dec", "pmra", "pmdec", "parallax")
    val columnsDouble = Seq("ra", "dec", "pmra", "pmdec", "parallax")
    var tgasDF = sqlContext.read.parquet(rutaDatosOriginales).select(columns.map(col): _*)
    columnsDouble.foreach { x =>
      tgasDF = tgasDF.withColumn(x, col(x).cast("Double"))
    }
    tgasDF.write.parquet(rutaFinal)
  }

  /**
   * GENERACION DEL DATAFRAME DE TGAS PARA EXPERIMENTO 2
   */
  def generateTGASExpDF2(sqlContext: SQLContext) = {
    val rutaDatosOriginales = "/user/dtizon/TGAS-DR1-parquet"
    val rutaFinal = "/user/dtizon/TGAS-Exp2"

    val columns = Seq("hip", "tycho2_id", "solution_id", "source_id", "ra", "dec", "pmra", "pmdec", "parallax")
    val columnsDouble = Seq("ra", "dec", "pmra", "pmdec", "parallax")
    var tgasDF = sqlContext.read.parquet(rutaDatosOriginales).select(columns.map(col): _*)
    columnsDouble.foreach { x =>
      tgasDF = tgasDF.withColumn(x, col(x).cast("Double"))
    }
    tgasDF.withColumn("X", Utils.getX(col("parallax"), col("ra"), col("dec")))
      .withColumn("Y", Utils.getY(col("parallax"), col("ra"), col("dec")))
      .withColumn("Z", Utils.getZ(col("parallax"), col("ra"), col("dec")))
      .withColumn("VTA", Utils.getVelocidadTransversal(col("parallax"), col("pmra")))
      .withColumn("VTD", Utils.getVelocidadTransversal(col("parallax"), col("pmdec")))
      .drop("ra").drop("dec").drop("parallax").drop("pmra").drop("pmdec")
      .write.parquet(rutaFinal)
  }

  def generateTychoFromHipExp1(dsExp1: Dataset[Experimento1], path: String) = {
    val pathIds = "C:/Users/Tornar/Dropbox/Master IA Avanzada/TFM/Hipparcos2Tycho.txt"
    import Spark.spark.implicits._
    val idsDF = Spark.spark.read.option("delimiter", "|").option("header", false).csv(pathIds).as[Ids]
    val TychoHipDF = idsDF.map { x => TychoHip(s"${x.Tycho1}-${x.Tycho2}-${x.Tycho3}", x.Hip) }
    val join = dsExp1.join(TychoHipDF, Seq("hip"), "left_outer").as[Experimento1TychoNew]
    val result = join.map(x => {
      val tycho2Old = x.tycho2_id
      val tycho2New = x.Tycho2New
      if (tycho2Old.isEmpty()) {
        Experimento1(x.hip, x.Tycho2New, x.ra, x.dec, x.pmra, x.pmdec, x.parallax)
      } else {
        Experimento1(x.hip, x.tycho2_id, x.ra, x.dec, x.pmra, x.pmdec, x.parallax)
      }
    })

    result.write.parquet(path)
  }

  case class Ids(var1: Double, var2: Double, var3: Double, Tycho1: String, Tycho2: String, Tycho3: String, Hip: String)
  case class TychoHip(Tycho2New: String, hip: String)
  case class Experimento1TychoNew(hip: String, tycho2_id: String, Tycho2New: String, ra: Double, dec: Double, pmra: Double, pmdec: Double, parallax: Double)
}