/* SimpleApp.scala */
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.rdd.RDD
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.Tokenizer
import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.feature.HashingTF
import org.apache.spark.mllib.feature.IDF

object SimpleApp {

	def main(args: Array[String]) {
		val productFile = "./data/products.txt"
		val conf = new SparkConf().setAppName("Simple Application")
		val sc = new SparkContext(conf)

		val tokenizer = new Tokenizer()

		// Create pair RDD of Product Name and word sequence
		val products: RDD[(String, Seq[String])] = sc.textFile("data/products_simple.txt").map(line => (line, line.toLowerCase.split(" ").toSeq))
		
		// Create pair RDD of Product Name and hashed term-frequency vector
		val hashingTF = new HashingTF()
		val tf: RDD[(String, Vector)] = products.mapValues(hashingTF.transform)
		tf.cache()
		
		// Create pair RDD of Product Name and tf-idf vector
		val idf = new IDF().fit(tf.values)
		val tfidf: RDD[(String, Vector)] = tf.mapValues(idf.transform)

		// Print 
		tfidf.foreach(println)
	}

	// def tokenize_text(line: String) {

	// }
}