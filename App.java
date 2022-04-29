package org.example.basicapp;

import org.apache.spark.*;
import org.apache.spark.sql.*;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.regression.RandomForestRegressor;
import org.apache.spark.mllib.evaluation.MulticlassMetrics;
import org.apache.spark.ml.feature.VectorAssembler;
import java.io.*;


public class App 
{  
  public static void main(String[] args) throws Exception
  {
    // Setup Spark Session
    SparkConf sparkConf = new SparkConf().setAppName("Training").setMaster("local")
      .set("spark.driver.host", "localhost").set("headers","true");
    SparkSession spark = SparkSession
      .builder()
      .config(sparkConf)
      .getOrCreate();

    // Read in the TrainingDataset and the ValidationDataset
    String trainingFilePath = "/home/ec2-user/myapp/src/main/java/org/example/basicapp/TrainingDataset.csv";
    Dataset<Row> df = spark.read().option("delimiter", ";").option("header", "true").format("csv").load(trainingFilePath);

    String validationFilePath = "/home/ec2-user/myapp/src/main/java/org/example/basicapp/ValidationDataset.csv";
    Dataset<Row> testData = spark.read().option("delimiter", ";").option("header", "true").format("csv").load(validationFilePath);

    // Format the data
    df = formatData(df);
    testData = formatData(testData);

    // Create Regressor
    RandomForestRegressor rf = new RandomForestRegressor()
      .setLabelCol("quality")
      .setFeaturesCol("features");

    // Create the Pipeline and fit the data to the model
    Pipeline pl1 = new Pipeline();
    pl1.setStages(new PipelineStage[]{rf});

    PipelineModel model1 = pl1.fit(df);

    // Make predictions
    Dataset<Row> predictions = model1.transform(testData);
    
    // Round the predicted values to match the quality
    predictions = predictions.withColumn("prediction", org.apache.spark.sql.functions.round(predictions.col("prediction")));
    
    // Select example rows to display.
    predictions.select("prediction", "quality", "features").show(10);

    // Get evaluation metrics.
    MulticlassMetrics metrics = new MulticlassMetrics(predictions.select("prediction", "quality"));

    // Overall statistics
    System.out.println("Accuracy = " + metrics.accuracy());

    // Stats by labels
    for (int i = 0; i < metrics.labels().length; i++) {
      System.out.format("Class %f precision = %f\n", metrics.labels()[i],metrics.precision(
        metrics.labels()[i]));
      System.out.format("Class %f recall = %f\n", metrics.labels()[i], metrics.recall(
        metrics.labels()[i]));
      System.out.format("Class %f F1 score = %f\n", metrics.labels()[i], metrics.fMeasure(
        metrics.labels()[i]));
    }

    //Weighted stats
    System.out.format("Weighted precision = %f\n", metrics.weightedPrecision());
    System.out.format("Weighted recall = %f\n", metrics.weightedRecall());
    System.out.format("Weighted F1 score = %f\n", metrics.weightedFMeasure());
    System.out.format("Weighted false positive rate = %f\n", metrics.weightedFalsePositiveRate());

    // Write the Model
    try 
    {
      model1.write().overwrite().save("/home/ec2-user/myapp/src/main/java/org/example/basicapp/TrainingModel");
    } 
    catch (IOException e) 
    {
      System.out.println("THE FILE TrainingModel COULD NOT BE WRITTEN.");
    }

    spark.stop();
  }

  public static Dataset<Row> formatData(Dataset<Row> df)
  {
    // Remove the extra "
    for (String col : df.columns()) 
    {
      df = df.withColumnRenamed(col, col.replace("\"",""));
    }

    df = df.na().drop().cache();

    // Cast String data into double
    for (String col : df.columns())
    {
      df = df.withColumn(col, df.col(col).cast("double"));
    }

    // Assemble all the columns (except for quality) into a single column
    String[] cols = {"alcohol", "sulphates", "pH", "density", "free sulfur dioxide", "total sulfur dioxide", "chlorides", "residual sugar", "citric acid", "volatile acidity", "fixed acidity"};
    VectorAssembler va = new VectorAssembler().setInputCols(cols).setOutputCol("features");

    df = va.transform(df).select("quality","features");

    //df.show(5);

    return df;
  }
}