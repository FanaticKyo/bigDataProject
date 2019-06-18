/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.daydream.projectspark;

import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.GBTClassificationModel;
import org.apache.spark.ml.classification.GBTClassifier;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.feature.VectorIndexer;
import org.apache.spark.ml.feature.VectorIndexerModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

/**
 *
 * @author shengxu
 */
public class GradientBoost {

    public static void main(String[] args) {
        SparkSession spark = SparkSession.builder().appName("Decision Tree").getOrCreate();
        // Load the data stored in LIBSVM format as a DataFrame.
        Dataset<Row> data = spark.read().option("header", "true").option("inferSchema", "true").csv("/home/shengxu/input/resampledData/*.csv");
        data = data.select("deviceCategory", "channelGrouping", "continent", "bounces", "isTrsacationRevenue");
        String[] features = new String[]{"deviceCategory", "channelGrouping", "continent"};
        for (String column : features) {
            StringIndexer indexer = new StringIndexer().setInputCol(column).setOutputCol(column + "Index");
            data = indexer.fit(data).transform(data);
        }
        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(new String[]{"deviceCategoryIndex", "channelGroupingIndex", "continentIndex", "bounces"})
                .setOutputCol("features");
        data = assembler.transform(data);

        VectorIndexerModel featureIndexer = new VectorIndexer().
                setInputCol("features")
                .setOutputCol("indexedFeatures")
                .setMaxCategories(10).fit(data); // features with > 4 distinct values are treated as continuous.

        Dataset<Row>[] splits = data.randomSplit(new double[]{0.7, 0.3});
        Dataset<Row> trainingData = splits[0];
        Dataset<Row> testData = splits[1];

        GBTClassifier gbt = new GBTClassifier()
                .setLabelCol("isTrsacationRevenue")
                .setFeaturesCol("indexedFeatures")
                .setMaxIter(10);
        Pipeline pipeline = new Pipeline()
                .setStages(new PipelineStage[]{featureIndexer, gbt});

// Train model. This also runs the indexers.
        PipelineModel model = pipeline.fit(trainingData);

// Make predictions.
        Dataset<Row> predictions = model.transform(testData);

// Select (prediction, true label) and compute test error.
        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("isTrsacationRevenue")
                .setPredictionCol("prediction")
                .setMetricName("accuracy");
        double accuracy = evaluator.evaluate(predictions);
        System.out.println("Test Error = " + (1.0 - accuracy));

        GBTClassificationModel gbtModel = (GBTClassificationModel) (model.stages()[1]);
        System.out.println("Learned classification GBT model:\n" + gbtModel.toDebugString());
        
        spark.stop();
    }
}
