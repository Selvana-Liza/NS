import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;

public class dl4jClassification {
    private static final int labelIndex = 50;       //сколько значений в каждой строке CSV-файла
    private static final int numClasses = 10;       //сколько классов в наборе данных
    private static final int batchSize = 133;       //сколько всего примеров
    private static final int numInputs = 50;        //колво-нейронов входной слой
    private static final int numHiddenLayers = 5;   //колво-нейронов скрытый слой
    private static final int numOutput = 10;        //колво-нейронов выходной слой
    private static final int nEpoch = 1000;         //колво эпох
    private static final long seed = 6;

    public static void main(String[] args) throws Exception {
        RecordReader recordReader = new CSVRecordReader(1, ',');
        recordReader.initialize(new FileSplit(
                    new ClassPathResource("setDL.csv").getFile()));

        //DataSetIterator управляет обходом набора данных и подготовкой данные для нейронной сети.
        DataSetIterator iterator = new RecordReaderDataSetIterator(recordReader,batchSize,labelIndex,numClasses);
        DataSet allData = iterator.next();
        //Перетасовать набор данных, чтобы избавиться от порядка классов в исходном файле
        allData.shuffle(123);
        //Разделяем выборку на тестовую и обучающую в соответсвии 75% на обучение
        SplitTestAndTrain testAndTrain = allData.splitTestAndTrain(0.75);
        //Получаем тестовую и обучающую выборки
        DataSet trainingData = testAndTrain.getTrain();
        DataSet testData = testAndTrain.getTest();

        //Нормализация данных
        DataNormalization normalizer = new NormalizerStandardize();
        //Собираем статистику (среднее/стандартное отклонение) из обучающих данных
        normalizer.fit(trainingData);
        //Применяем нормализацию к обучающим данным
        normalizer.transform(trainingData);
        //Применяем нормализацию к тестовым данным
        normalizer.transform(testData);


        //ПОСТРОЕНИЕ МОДЕЛИ
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .activation(Activation.TANH)
                .weightInit(WeightInit.XAVIER)
                .updater(new Sgd(0.1))
                .l2(1e-4)
                .list()
                .layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(numHiddenLayers)
                        .build())
                .layer(1, new DenseLayer.Builder().nIn(numHiddenLayers).nOut(numHiddenLayers)
                        .build())
                .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        //Переопределить глобальную активацию TANH с помощью softmax для этого слоя
                        .activation(Activation.SOFTMAX)
                        .nIn(numHiddenLayers).nOut(numOutput).build())
                .build();

        //Инициализация модели
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        //Записываем оценку один раз каждые 100 итераций
        model.setListeners(new ScoreIterationListener(100));

        for(int i=0; i<nEpoch; i++ ) {
            model.fit(trainingData);
        }

        //Сохранение модели в файл
        model.save(new File("dl4j.model"));

        getEvaluation(model,testData);

        System.out.print("---------------------------------------------------------------------------\n");
        System.out.print("**************ЗАГРУЗКА МОДЕЛИ ИЗ ФАЙЛА************************\n");
        //Загрузка модели из файла
        MultiLayerNetwork modelLoad = MultiLayerNetwork.load(new File("dl4j.model"),true);

        getEvaluation(modelLoad,testData);
    }


    //Метод для вывода результата обучения
    public static void getEvaluation(MultiLayerNetwork model, DataSet testData){
        //Оцениваем модель на тестовом наборе
        Evaluation eval = new Evaluation(10);
        INDArray output = model.output(testData.getFeatures());
        eval.eval(testData.getLabels(), output);

        //Вывод оценки модели
        System.out.println(eval.stats()); //confusion matrix, evaluation metrics
        System.out.println("-----------------------------------");
        System.out.println(eval.confusionToString());
        System.out.println("-----------------------------------");
        System.out.println(output);
        System.out.println("-----------------------------------");

        //Вывод результата по каждому экземпляру данных
        for(int i = 0;i<output.rows();i++) {
            String actual = String.valueOf(testData.get(i).outcome());
            String prediction = String.valueOf(output.getRow(i).argMax());
            System.out.printf((i+1) + ". Actual: " + actual + "  Predicted: " + prediction);
            System.out.println(!prediction.equals(actual) ? " *" : "");
        }
    }
}
