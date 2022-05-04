import java.awt.*;
import java.io.File;
import java.util.Random;

import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.lazy.IBk;
import weka.classifiers.trees.J48;
import weka.core.CapabilitiesIgnorer;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.CSVLoader;
import weka.gui.treevisualizer.PlaceNode2;
import weka.gui.treevisualizer.TreeVisualizer;

import javax.swing.*;

public class Classification {

    public static void main(String[] args) throws Exception {
        String path = "set.csv";
        Instances data = loadDataSet(path);
        data.setClassIndex(data.numAttributes() - 1);
        data.randomize(new Random(123));

        Instances trainSet = new Instances(data, 0, 100);
        Instances testSet = new Instances(data, 100, 33);

        //Standardize
        /*Standardize filter = new Standardize();
        filter.setInputFormat(train);  // initializing the filter once with training set
        Instances trainSet = Filter.useFilter(trainSet, filter);  // configures the Filter based on train instances and returns filtered instances
        Instances testSet = Filter.useFilter(testSet, filter);    // create new test set*/

        TreeClassifierBuild(trainSet,testSet);
        NaiveBayesClassifierBuild(trainSet,testSet);
        kNNClassifierBuild(trainSet,testSet);

        System.out.print("---------------------------------------------------------------------------\n");
        System.out.print("**************ЗАГРУЗКА МОДЕЛИ ИЗ ФАЙЛА************************\n");

        Classifier classifier = (Classifier) loadModel("IBk1.model");
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

        System.out.print("---------------------------------------------------------------------------\n");
        System.out.print("***************************************************************************\n");

        //Cross-validation
        /*Evaluation eval = new Evaluation(data);
        eval.crossValidateModel(classifier4, data, 10, new Random(1));*/
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

    private static void kNNClassifierBuild(Instances trainSet, Instances testSet) throws Exception {
        System.out.print("-----------------------------kNN-CLASSIFIER--------------------------------\n");

        Evaluation eval = new Evaluation(trainSet);
        IBk classifier = new IBk();
        classifier.setOptions(weka.core.Utils.splitOptions("-K 1"));
        classifier.buildClassifier(trainSet);
        System.out.print(classifier.toString());

        saveModel("IBk1.model",classifier);

        eval.evaluateModel(classifier, testSet);
        System.out.println(eval.toSummaryString());


        System.out.print("*******************************************************************\n");
        int k = 0;
        for (int j = 0; j < testSet.numInstances(); ++j) {
            double res = classifier.classifyInstance(testSet.get(j));
            //System.out.print(testSet.classAttribute().value((int) res) + "\n");
            String actual = testSet.classAttribute().value((int) testSet.instance(j).classValue());
            String prediction = testSet.classAttribute().value((int) res);
            System.out.printf(++k + ". \tActual: " + actual + "\t  Predicted: " + prediction);
            System.out.println(!prediction.equals(actual) ? " *" : "");
        }
    }

    private static void TreeClassifierBuild(Instances trainSet, Instances testSet) throws Exception {
        System.out.print("-----------------------------TREE-CLASSIFIER-------------------------------\n");

        Evaluation eval = new Evaluation(trainSet);
        J48 classifier = new J48();
        classifier.setOptions(Utils.splitOptions("-C 0.1"));
        classifier.buildClassifier(trainSet);
        System.out.print(classifier.toString());

        saveModel("J48_1.model",classifier);

        eval.evaluateModel(classifier, testSet);
        System.out.println(eval.toSummaryString());

        System.out.print("*******************************************************************\n");
        int k = 0;
        for (int j = 0; j < testSet.numInstances(); ++j) {
            double res = classifier.classifyInstance(testSet.get(j));
            //System.out.print(testSet.classAttribute().value((int) res) + "\n");
            String actual = testSet.classAttribute().value((int) testSet.instance(j).classValue());
            String prediction = testSet.classAttribute().value((int) res);
            System.out.printf(++k + ". \tActual: " + actual + "\t  Predicted: " + prediction);
            System.out.println(!prediction.equals(actual) ? " *" : "");
        }
        //visualizeTree(classifier);
    }

    private static void NaiveBayesClassifierBuild(Instances trainSet, Instances testSet) throws Exception {
        System.out.print("-------------------------NAIVE-BAYES-CLASSIFIER----------------------------\n");

        Evaluation eval = new Evaluation(trainSet);
        NaiveBayes classifier = new NaiveBayes();
        classifier.buildClassifier(trainSet);
        System.out.print(classifier.toString());

        saveModel("NaiveBayes1.model",classifier);

        eval.evaluateModel(classifier, testSet);
        System.out.println(eval.toSummaryString());

        System.out.print("*******************************************************************\n");
        int k = 0;
        for (int j = 0; j < testSet.numInstances(); ++j) {
            double res = classifier.classifyInstance(testSet.get(j));
            //System.out.print(testSet.classAttribute().value((int) res) + "\n");
            String actual = testSet.classAttribute().value((int) testSet.instance(j).classValue());
            String prediction = testSet.classAttribute().value((int) res);
            System.out.printf(++k + ". \tActual: " + actual + " \tPredicted: " + prediction);
            System.out.println(!prediction.equals(actual) ? "\t *" : "");
        }
    }
}
