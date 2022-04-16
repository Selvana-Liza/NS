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

import java.io.IOException;
import java.util.concurrent.TimeUnit;

public class dl4jClassification {
    public static void main(String[] args) throws Exception {
        RecordReader recordReader = new CSVRecordReader(1, ',');
        recordReader.initialize(new FileSplit(
                    new ClassPathResource("setDL.csv").getFile()));

        int labelIndex = 50;     //5 значений в каждой строке CSV-файла
        int numClasses = 10;     //3 класса (типы цветов ириса) в наборе данных ириса.
        int batchSize = 6;    //Набор данных Iris: всего 150 примеров.

        //DataSetIterator управляет обходом набора данных и подготовкой данные для нейронной сети.
        DataSetIterator iterator = new RecordReaderDataSetIterator(recordReader,batchSize,labelIndex,numClasses);
        DataSet allData = iterator.next();
        //перетасовать набор данных, чтобы избавиться от порядка классов в исходном файле
        allData.shuffle(42);
        //Разделяем выборку на тестовую и обучающую в соответсвии 75% на обучение
        SplitTestAndTrain testAndTrain = allData.splitTestAndTrain(0.75);
        //получаем выборки
        DataSet trainingData = testAndTrain.getTrain();
        DataSet testData = testAndTrain.getTest();

        //нормализация данных
        DataNormalization normalizer = new NormalizerStandardize();
        //Собираем статистику (среднее/стандартное отклонение) из обучающих данных
        normalizer.fit(trainingData);
        // Применяем нормализацию к обучающим данным
        normalizer.transform(trainingData);
        // Применяем нормализацию к тестовым данным
        normalizer.transform(testData);

        final int numInputs = 50;  //колво-нейронов входной слой
        int outputNum = 10;  //колво-нейронов выходной слой
        long seed = 6;

        //ПОСТРОЕНИЕ МОДЕЛИ
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .activation(Activation.TANH)
                .weightInit(WeightInit.XAVIER)
                .updater(new Sgd(0.1))
                .l2(1e-4)
                .list()
                .layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(3)
                        .build())
                .layer(1, new DenseLayer.Builder().nIn(3).nOut(3)
                        .build())
                .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
        //Переопределить глобальную активацию TANH с помощью softmax для этого слоя
                        .activation(Activation.SOFTMAX)
                        .nIn(3).nOut(outputNum).build())
                .build();

        //Инициализация модели
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        // записываем оценку один раз каждые 100 итераций
        model.setListeners(new ScoreIterationListener(100));

        for(int i=0; i<1000; i++ ) {
            model.fit(trainingData);
        }
               //оцениваем модель на тестовом наборе
        Evaluation eval = new Evaluation(10);
        INDArray output = model.output(testData.getFeatures());
        eval.eval(testData.getLabels(), output);


        System.out.println(eval.stats()); //confusion matrix, evaluation metrics
        System.out.println("-----------------------------------");
        System.out.println(eval.getLabelsList());//список классов
        System.out.println("-----------------------------------");
        System.out.println(eval.confusionToString());
        System.out.println("-----------------------------------");
        System.out.println(output);

    }
}
