package clustering.metrics

import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession

object Spark {  
  var conf = new SparkConf()    
  lazy val spark = SparkSession.builder().config(conf).getOrCreate()
}