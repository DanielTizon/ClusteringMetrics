package clustering.metrics

import scala.reflect.runtime.universe

import org.apache.spark.ml.feature.StandardScaler
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.functions.udf

object Utils {

  def standarize(vectorData: DataFrame): DataFrame = {
    val scaler = new StandardScaler().setInputCol("features").setOutputCol("scaledFeatures").setWithStd(true).setWithMean(true)
    val scalerModel = scaler.fit(vectorData)
    scalerModel.transform(vectorData).drop("features").withColumnRenamed("scaledFeatures", "features")
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
}