package njit.sarah.wineproject;

import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.classification.RandomForestClassifier;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import static org.apache.spark.sql.functions.col;
import static org.apache.spark.sql.functions.when;

public class WineQualityTraining {

    public static void main(String[] args) {
        // Initialize Spark session
        SparkSession spark = SparkSession.builder().appName("QualityTrainingForWine").getOrCreate();

        // Schema definition for CSV data
        StructType csvSchema = new StructType()
                .add("fixed_acidity", DataTypes.DoubleType)
                .add("volatile_acidity", DataTypes.DoubleType)
                .add("citric_acid", DataTypes.DoubleType)
                .add("residual_sugar", DataTypes.DoubleType)
                .add("chlorides", DataTypes.DoubleType)
                .add("free_sulfur_dioxide", DataTypes.DoubleType)
                .add("total_sulfur_dioxide", DataTypes.DoubleType)
                .add("density", DataTypes.DoubleType)
                .add("pH", DataTypes.DoubleType)
                .add("sulphates", DataTypes.DoubleType)
                .add("alcohol", DataTypes.DoubleType)
                .add("quality", DataTypes.DoubleType);

        // Reading and processing the dataset
        Dataset<Row> dataset = spark.read().format("csv").schema(csvSchema)
                .option("header", "true").option("delimiter", ";").option("quote", "\"")
                .option("ignoreLeadingWhiteSpace", true).option("ignoreTrailingWhiteSpace", true)
                .load("file:///home/ec2-user/WineQualityPrediction/TrainingDataset.csv");

        for (String columnName : dataset.columns()) {
            dataset = dataset.withColumn(columnName, functions.trim(col(columnName)));
        }

        dataset = dataset.withColumn("quality", when(col("quality").gt(7), 1).otherwise(0));

        // Feature vector preparation
        String[] featureCols = dataset.columns();
        featureCols = Arrays.copyOfRange(featureCols, 0, featureCols.length - 1);
        VectorAssembler vectorAssembler = new VectorAssembler()
                .setInputCols(featureCols)
                .setOutputCol("features");
        dataset = vectorAssembler.transform(dataset);

        // Splitting the dataset
        Dataset<Row>[] splits = dataset.randomSplit(new double[]{0.8, 0.2});
        Dataset<Row> trainData = splits[0];
        Dataset<Row> testData = splits[1];

        // Model training with Random Forest Classifier
        RandomForestClassifier randomForest = new RandomForestClassifier()
                .setLabelCol("quality")
                .setFeaturesCol("features")
                .setNumTrees(200);

        Pipeline pipeline = new Pipeline()
                .setStages(new RandomForestClassifier[]{randomForest});

        PipelineModel trainedModel = pipeline.fit(trainData);

        // Model prediction and evaluation
        Dataset<Row> testPredictions = trainedModel.transform(testData);
        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("quality")
                .setPredictionCol("prediction")
                .setMetricName("f1");
        double f1Metric = evaluator.evaluate(testPredictions);
        System.out.println("Evaluated F1 Score: " + f1Metric);

        // Saving the model
        trainedModel.write().overwrite().save("file:///home/ec2-user/WineQualityPrediction/qualitytrainingforwine");
    }
}
