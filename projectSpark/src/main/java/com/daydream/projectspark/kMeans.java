/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.daydream.projectspark;

import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.feature.VectorIndexer;
import org.apache.spark.ml.feature.VectorIndexerModel;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.ml.clustering.KMeansModel;
import org.apache.spark.ml.clustering.KMeans;
import org.apache.spark.ml.evaluation.ClusteringEvaluator;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;

/**
 *
 * @author shengxu
 */
public class kMeans {

    public static void main(String[] args) {

        SparkSession spark = SparkSession.builder().appName("Decision Tree").getOrCreate();
        // Load the data stored in LIBSVM format as a DataFrame.
        Dataset<Row> data = spark.read().option("header", "true").option("inferSchema", "true").csv("/home/shengxu/input/resampledData/*.csv");
        data = data.select("deviceCategory", "channelGrouping", "continent", "bounces");
        String[] features = new String[]{"deviceCategory", "channelGrouping", "continent"};
        for (String column : features) {
            StringIndexer indexer = new StringIndexer().setInputCol(column).setOutputCol(column + "Index");
            data = indexer.fit(data).transform(data);
        }
        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(new String[]{"deviceCategoryIndex", "channelGroupingIndex", "continentIndex", "bounces"})
                .setOutputCol("features");
        data = assembler.transform(data).select("features");

        KMeans kmeans = new KMeans().setK(6).setSeed(1L);
        KMeansModel model = kmeans.fit(data);

// Make predictions
        Dataset<Row> predictions = model.transform(data);

// Evaluate clustering by computing Silhouette score
        ClusteringEvaluator evaluator = new ClusteringEvaluator();

        double silhouette = evaluator.evaluate(predictions);
        double WSSSE = model.computeCost(data);
        
        System.out.println("Within Set Sum of Squared Errors = " + WSSSE);
        System.out.println(
                "Silhouette with squared euclidean distance = " + silhouette);

// Shows the result.
        Vector[] centers = model.clusterCenters();

        System.out.println(
                "Cluster Centers: ");
        for (Vector center
                : centers) {
            System.out.println(center);
        }
        spark.stop();
    }

}
