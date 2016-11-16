package clustering.metrics

import org.apache.spark.sql.SparkSession
import org.apache.spark.SparkConf

object Spark {  
  var conf = new SparkConf()    
  lazy val spark = SparkSession.builder().config(conf).getOrCreate()
}