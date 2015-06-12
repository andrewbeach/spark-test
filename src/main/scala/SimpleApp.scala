/* SimpleApp.scala */
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.rdd.RDD

import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.feature.HashingTF
import org.apache.spark.mllib.feature.IDF

object SimpleApp {

	def main(args: Array[String]) {
		val productFile = "./data/products.txt"
		val conf = new SparkConf().setAppName("Simple Application")
		val sc = new SparkContext(conf)

		// Create pair RDD of Product Name and word sequence
		val products: RDD[(String, Seq[String])] = sc.textFile(productFile).map(line => (line, line.toLowerCase.split(" ").toSeq))
		
		// Create pair RDD of Product Name and hashed term-frequency vector
		val hashingTF = new HashingTF(numFeatures = 10000)
		val tf: RDD[(String, Vector)] = products.mapValues(hashingTF.transform)
		tf.cache()
		
		// Create pair RDD of Product Name and tf-idf vector
		val idf = new IDF().fit(tf.values)
		val tfidf: RDD[(String, Vector)] = tf.mapValues(idf.transform)
		tfidf.values.cache()

		// Cluster the data into classes using KMeans
		val kmeans = new KMeans().setK(200)
		val clusters: KMeansModel = kmeans.run(tfidf.values)

		// Print 
		val product_points: RDD[(String, Int)] = tfidf.mapValues(clusters.predict)

		val product_clusters: RDD[(Int, String)] = product_points.map(_.swap)

		val grouped_products = product_clusters.groupByKey()

		grouped_products.foreach(println)



	}

	// def tokenize_text(line: String) {

	// }
}