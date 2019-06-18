/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.daydream.projectspark;

import org.apache.spark.ml.clustering.GaussianMixture;
import org.apache.spark.ml.clustering.GaussianMixtureModel;
import org.apache.spark.ml.evaluation.ClusteringEvaluator;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

/**
 *
 * @author shengxu
 */
public class GMM {

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

        GaussianMixture gmm = new GaussianMixture().setK(5);
        GaussianMixtureModel model = gmm.fit(data);
        
        Dataset<Row> predictions = model.transform(data);
        
        ClusteringEvaluator evaluator = new ClusteringEvaluator();
        double silhouette = evaluator.evaluate(predictions);
        System.out.println(
                "Silhouette with squared euclidean distance = " + silhouette);

// Output the parameters of the mixture model
        for (int i = 0; i < model.getK(); i++) {
            System.out.printf("Gaussian %d:\nweight=%f\nmu=%s\nsigma=\n%s\n\n",
                    i, model.weights()[i], model.gaussians()[i].mean(), model.gaussians()[i].cov());

        }
        spark.stop();
    }
}
