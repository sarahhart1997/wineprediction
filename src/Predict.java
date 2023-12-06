import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import static org.apache.spark.sql.functions.col;
import static org.apache.spark.sql.functions.when;

import java.io.File;

import spark.Request;
import spark.Response;
import spark.Route;

package njit.sarah.wineproject; 

import static spark.Spark.*;

public class WineQualityPrediction {

    public static void main(String[] args) {
        SparkSession spark = SparkSession.builder().appName("QualityInferenceForWine").getOrCreate();

        // Load the RandomForestClassificationModel
        PipelineModel rfModel = PipelineModel.load("/src/qualitytrainingforwine");

        // Define schema for incoming CSV data
        StructType dataSchema = new StructType()
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

        post("/predict", (request, response) -> makePrediction(request, response, spark, rfModel, dataSchema));
    }

    private static String makePrediction(Request request, Response response, SparkSession spark, PipelineModel rfModel, StructType dataSchema) {
        // Receive and save the uploaded file
        File uploadedFile = request.raw().getPart("file").getSubmittedFile();
        String tmpDir = "/tmp";
        File tmpDirFile = new File(tmpDir);
        tmpDirFile.mkdirs();
        String filePath = tmpDir + File.separator + uploadedFile.getName();
        uploadedFile.renameTo(new File(filePath));

        // Process the dataset
        Dataset<Row> validationDataset = spark.read().format("csv").schema(dataSchema)
                .option("header", "true").option("delimiter", ";").load(filePath);
        validationDataset = validationDataset.withColumn("quality", when(col("quality").gt(7), 1).otherwise(0));

        // Feature vectorization
        VectorAssembler featuresAssembler = new VectorAssembler()
                .setInputCols(validationDataset.columns())
                .setOutputCol("features");
        validationDataset = featuresAssembler.transform(validationDataset);

        // Predict using the model
        Dataset<Row> predictionResults = rfModel.transform(validationDataset);

        // Model evaluation
        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("quality")
                .setPredictionCol("prediction")
                .setMetricName("f1");
        double f1Metric = evaluator.evaluate(predictionResults);

        // Return JSON response
        response.type("application/json");
        return "{\"f1_score\": " + f1Metric + "}";
    }
}
