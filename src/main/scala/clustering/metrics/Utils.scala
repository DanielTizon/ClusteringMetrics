package clustering.metrics

import scala.reflect.runtime.universe

import org.apache.spark.ml.feature.StandardScaler
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.functions.udf

object Utils {
  
  def removeOutliers(vectorData: DataFrame, factor: Int): DataFrame = {
    vectorData.filter(x => {
      val data = x.getAs[org.apache.spark.ml.linalg.Vector]("features")
      var isOutlier = false
      for (i <- 0 to data.size-1) {
        if (data(i) > factor || data(i) < -factor) isOutlier = true
      }
      !isOutlier
    })
  }

  def standarize(vectorData: DataFrame): DataFrame = {
    val scaler = new StandardScaler().setInputCol("features").setOutputCol("scaledFeatures").setWithStd(true).setWithMean(true)
    val scalerModel = scaler.fit(vectorData)
    scalerModel.transform(vectorData).drop("features").withColumnRenamed("scaledFeatures", "features")
  }

  val getColor = udf((intValue: Int) => {
    if (intValue == 0) "gray"
    else if (intValue == 1) "lightcoral"
    else if (intValue == 2) "tan"
    else if (intValue == 3) "g"
    else if (intValue == 4) "c"
    else if (intValue == 5) "m"
    else if (intValue == 6) "plum"
    else if (intValue == 7) "b"
    else if (intValue == 8) "lightblue"
    else if (intValue == 9) "steelblue"
    else if (intValue == 10) "turquoise"
    else if (intValue == 11) "pink"
    else if (intValue == 12) "silver"
    else if (intValue == 13) "mediumseagreen"
    else if (intValue == 14) "y"
    else "r"
  })

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
    tgasDF.withColumn("X", getX(col("parallax"), col("ra"), col("dec")))
      .withColumn("Y", getY(col("parallax"), col("ra"), col("dec")))
      .withColumn("Z", getZ(col("parallax"), col("ra"), col("dec")))
      .withColumn("VTA", getVelocidadTransversal(col("parallax"), col("pmra")))
      .withColumn("VTD", getVelocidadTransversal(col("parallax"), col("pmdec")))
      .drop("ra").drop("dec").drop("parallax").drop("pmra").drop("pmdec")
      .write.parquet(rutaFinal)
  }
}