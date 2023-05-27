// AngelSparkSession的引入
import com.tencent.angel.sona.core.{AngelSparkConf, AngelSparkSession}
import com.tencent.angel.sona.data._
import com.tencent.angel.sona.ml.evaluation.{Evaluator, MulticlassClassificationEvaluator}
import com.tencent.angel.sona.ml.feature._
import com.tencent.angel.sona.ml.param.shared.HasWeightCol
import com.tencent.angel.sona.ml.{Pipeline, PipelineModel}
import com.tencent.angel.sona.ml.tuning.{CrossValidator, ParamGridBuilder}
import com.tencent.angel.sona.source.libsvm._
import com.tencent.angel.sona.source.libsvm.LibSVMFileFormat

object SparkOnAngelPipeline {

  def main(args: Array[String]): Unit = {
    val inputPath = "/Users/anya/Downloads/dd/欺诈分析/creditcard2.csv"
    val nativeModelPath = "/Users/anya/Downloads/sparkonangel/nativeModel"

    // AngelSparkSession的创建
    val conf = new AngelSparkConf()
      .set("angel.workergroup.group.num", "1")
      .set("angel.deploy.mode", "LOCAL")

    val spark = AngelSparkSession.builder()
      .appName("XGBoost4J-Spark Pipeline Example")
      .sparkConf(conf)
      .getOrCreate()
    // 数据加载为Angel的Dataframe（AngelDataFrame）
    // 加载原始数据集
    val rawInput = spark.read.format(classOf[LibSVMFileFormat].getName)
      .load(inputPath)
    //    Angel的VectorAssembler
    val assembler = new AngelVectorAssembler()
      .setInputCols(Array("Time", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10", "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20", "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28", "Amount"))
      .setOutputCol("features")
    // 标签索引器
    val labelIndexer = new AngelStringIndexer()
      .setInputCol("Class")
      .setOutputCol("ClassIndex")
      .fit(rawInput)
    // 创建为AngelXGBoostClassifier
    val booster = new AngelXGBoostClassifier(
      Map("eta" -> 0.1f, "max_depth" -> 2, "objective" -> "multi:softprob", "num_class" -> 2, "num_round" -> 5)
    )
    booster.setFeaturesCol("features")
    booster.setLabelCol("ClassIndex")
    // Angel的索引转换器
    val labelConverter = new AngelIndexToString()
      .setInputCol("prediction")
      .setOutputCol("realLabel")
      .setLabels(labelIndexer.labels)
    // Angel创建管道
    val pipeline = new AngelPipeline()
      .setStages(Array(assembler, labelIndexer, booster, labelConverter))
    // 在原始数据上拟合管道模型
    val model = pipeline.fit(rawInput)
    // 对原始数据进行预测
    val prediction = model.transform(rawInput)
    prediction.show(false)
    // 评估模型准确率
    val evaluator = new AngelMulticlassClassificationEvaluator()
      .setLabelCol("ClassIndex")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")

    val accuracy = evaluator.evaluate(prediction)
    println("The model accuracy is : " + accuracy)
    // 设置参数网格进行模型调优
    val paramGrid = new AngelParamGridBuilder()
      .addGrid(booster.maxDepth, Array(3, 8))
      .addGrid(booster.eta, Array(0.2, 0.6))
      .build()
    // 创建交叉验证器
    val cv = new AngelCrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(3)
    // 对整个数据集进行交叉验证
    val cvModel = cv.fit(rawInput)
    // 获取最佳模型
    val bestModel = cvModel.bestModel.asInstanceOf[AngelPipelineModel].stages(2)
      .asInstanceOf[AngelXGBoostClassificationModel]

    println("The params of best XGBoostClassification model : " + bestModel.extractParamMap())
    println("The training summary of best XGBoostClassificationModel : " + bestModel.summary)
    // 保存本地XGBoost模型
    bestModel.nativeBooster.saveModel(nativeModelPath)


    spark.stop()
  }
}
