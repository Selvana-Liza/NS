import java.awt.*;
import java.io.File;
import java.util.Random;

import org.nd4j.linalg.api.ops.impl.transforms.custom.Standardize;
import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.lazy.IBk;
import weka.classifiers.trees.J48;
import weka.core.CapabilitiesIgnorer;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.CSVLoader;
import weka.filters.Filter;
import weka.gui.treevisualizer.PlaceNode2;
import weka.gui.treevisualizer.TreeVisualizer;

import javax.swing.*;

public class Classification {

    private static Instances trainSet;
    private static Instances testSet;
    private static Instances data;

    public static void main(String[] args) throws Exception {
        String path = "set.csv";
        data = loadDataSet(path);
        data.setClassIndex(data.numAttributes() - 1);
        data.randomize(new Random(123));

        trainSet = new Instances(data, 0, 100);
        testSet = new Instances(data, 100, 33);

        TreeClassifierBuild();
        NaiveBayesClassifierBuild();
        kNNClassifierBuild();

        System.out.print("\n---------------------------------------------------------------------------\n");
        System.out.println("**************ЗАГРУЗКА МОДЕЛИ ИЗ ФАЙЛА************************\n");

        Classifier classifier = (Classifier) loadModel("IBk1.model");
        System.out.print(classifier.toString());
        getEvaluation(classifier);

        System.out.print("---------------------------------------------------------------------------\n");
        System.out.println("***************************************************************************\n");
    }

    public static void saveModel(String modelFile, Classifier classifier) throws Exception {
        weka.core.SerializationHelper.write(modelFile, classifier);
    }

    public static CapabilitiesIgnorer loadModel(String modelFile) throws Exception {
        Classifier classifier = (Classifier) weka.core.SerializationHelper.read(modelFile);
                return (CapabilitiesIgnorer) classifier;
    }

    public static Instances loadDataSet(String path) throws Exception {
        CSVLoader loader = new CSVLoader();
        loader.setSource(new File(path));
        Instances data = loader.getDataSet();
        data.setClassIndex(data.numAttributes() - 1);

        /*DataSource source = new DataSource(path);
        Instances data = source.getDataSet();
        if (data.classIndex() == -1) {
            data.setClassIndex(data.numAttributes() - 1);
        }*/
        return data;
    }

    public static void getEvaluation(Classifier classifier) throws Exception {
        Evaluation eval = new Evaluation(trainSet);
        eval.evaluateModel(classifier, testSet);
        System.out.println(eval.toSummaryString());

        int k = 0;
        for (int j = 0; j < testSet.numInstances(); ++j) {
            double res = classifier.classifyInstance(testSet.get(j));
            String actual = testSet.classAttribute().value((int) testSet.instance(j).classValue());
            String prediction = testSet.classAttribute().value((int) res);
            System.out.printf(++k + ". \tActual: " + actual + " \tPredicted: " + prediction);
            System.out.println(!prediction.equals(actual) ? "\t *" : "");
        }
    }

    public static void getCrossValidation(Classifier classifier) throws Exception {
        //Cross-validation
        Evaluation eval = new Evaluation(data);
        eval.crossValidateModel(classifier, data, 10, new Random(1));
    }

    private static void kNNClassifierBuild() throws Exception {
        System.out.println("\n-----------------------------kNN-CLASSIFIER--------------------------------\n");

        IBk classifier = new IBk();
        classifier.setOptions(weka.core.Utils.splitOptions("-K 1"));
        classifier.buildClassifier(trainSet);
        System.out.print(classifier.toString());

        saveModel("IBk1.model",classifier);
        getEvaluation(classifier);
    }

    private static void TreeClassifierBuild() throws Exception {
        System.out.println("\n-----------------------------TREE-CLASSIFIER-------------------------------\n");

        J48 classifier = new J48();
        classifier.setOptions(Utils.splitOptions("-C 0.1"));
        classifier.buildClassifier(trainSet);
        System.out.print(classifier.toString());

        saveModel("J48_1.model",classifier);
        getEvaluation(classifier);
    }

    private static void NaiveBayesClassifierBuild() throws Exception {
        System.out.println("\n-------------------------NAIVE-BAYES-CLASSIFIER----------------------------\n");

        NaiveBayes classifier = new NaiveBayes();
        classifier.buildClassifier(trainSet);
        System.out.print(classifier.toString());

        saveModel("NaiveBayes1.model",classifier);
        getEvaluation(classifier);
    }
}
