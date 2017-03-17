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
      for (i <- 0 to data.size - 1) {
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
    val colors = "slateblue,sienna,deepskyblue,darkcyan,orchid,dimgray,lightgreen,thistle,firebrick,mediumaquamarine,silver,red,lightseagreen,yellow,darkred,seashell,olivedrab," +
      "mediumvioletred,mistyrose,darkolivegreen,slategray,tomato,mediumorchid,darkgrey,chocolate,purple,navajowhite,darkorange,rosybrown,mediumseagreen,yellowgreen,fuchsia,darkblue,darkgoldenrod,palegreen,forestgreen," +
      "aliceblue,greenyellow,darkkhaki,orangered,maroon,aquamarine,darkslateblue,deeppink,sage,paleturquoise,lawngreen,aqua,skyblue,peachpuff,beige,gainsboro,peru,brown,black,lightgrey,navy,plum,lemonchiffon,lightsage," +
      "mediumturquoise,darkslategray,indianred,indigo,lightyellow,slategrey,turquoise,blanchedalmond,antiquewhite,lightblue,lightcyan,azure,grey,lightsteelblue,lightgoldenrodyellow,crimson,cyan,mintcream,lavender," +
      "ghostwhite,palegoldenrod,cornsilk,darkgray,lightskyblue,dodgerblue,mediumslateblue,snow,limegreen,honeydew,goldenrod,blue,cadetblue,darksalmon,bisque,white,lavenderblush,wheat,mediumspringgreen,mediumblue," +
      "palevioletred,lightcoral,whitesmoke,darksage,khaki,mediumpurple,orange,floralwhite,cornflowerblue,darkmagenta,pink,midnightblue,darkturquoise,lime,burlywood,sandybrown,olive,oldlace,darkseagreen,gray,coral," +
      "darkgreen,linen,saddlebrown,lightpink,gold,hotpink,chartreuse,violet,blueviolet,royalblue,green,papayawhip,dimgrey,tan,darkviolet,springgreen,lightsalmon,steelblue,seagreen,darkorchid,lightgray,salmon,magenta," +
      "moccasin,teal,darkslategrey,lightslategray,lightslategrey,ivory,powderblue"

    val colorsList = colors.split(",")

    colorsList(intValue)
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