import java.awt.*;
import java.io.File;
import java.util.List;
import java.util.Random;

import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.evaluation.Prediction;
import weka.classifiers.lazy.IBk;
import weka.classifiers.trees.J48;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.CSVLoader;
import weka.core.converters.ConverterUtils.DataSource;
import weka.gui.treevisualizer.PlaceNode2;
import weka.gui.treevisualizer.TreeVisualizer;

import javax.swing.*;

public class Classification {

    public static void main(String[] args) throws Exception {
        String path = "set.csv";
        Instances data = loadDataSet(path);
        data.setClassIndex(data.numAttributes() - 1);
        data.randomize(new Random(0L));

        Instances trainSet = new Instances(data, 0, 4);
        Instances testSet = new Instances(data, 4, 2);

        //Standardize
        /*Standardize filter = new Standardize();
        filter.setInputFormat(train);  // initializing the filter once with training set
        Instances trainset = Filter.useFilter(trainset, filter);  // configures the Filter based on train instances and returns filtered instances
        Instances testset = Filter.useFilter(testset, filter);    // create new test set*/

        TreeClassifierBuild(trainSet,testSet);
        NaiveBayesClassifierBuild(trainSet,testSet);
        kNNClassifierBuild(trainSet,testSet);

        System.out.print("---------------------------------------------------------------------------\n");

        //Cross-validation
        /*Evaluation eval = new Evaluation(data);
        eval.crossValidateModel(classifier4, data, 10, new Random(1));*/
    }

    public static Instances loadDataSet(String path) throws Exception {
        /*CSVLoader loader = new CSVLoader();
        loader.setSource(new File(path));
        Instances data = loader.getDataSet();*/

        DataSource source = new DataSource(path);
        Instances data = source.getDataSet();
        if (data.classIndex() == -1) {
            data.setClassIndex(data.numAttributes() - 1);
        }
        return data;
    }

    private static void kNNClassifierBuild(Instances trainSet, Instances testSet) throws Exception {
        System.out.print("-----------------------------kNN-CLASSIFIER--------------------------------\n");

        Evaluation Test = new Evaluation(trainSet);
        IBk classifier = new IBk();
        classifier.setOptions(weka.core.Utils.splitOptions("-K 1"));
        classifier.buildClassifier(trainSet);
        System.out.print(classifier.toString());

        Test.evaluateModel(classifier, testSet);
        System.out.println(Test.toSummaryString());


        System.out.print("*******************************************************************\n");
        for (int j = 0; j < testSet.numInstances(); ++j) {
            double res = classifier.classifyInstance(testSet.get(j));
            //System.out.print(testSet.classAttribute().value((int) res) + "\n");
            System.out.print("Actual: " + testSet.classAttribute().value((int) testSet.instance(j).classValue()));
            System.out.println(", predicted: " + testSet.classAttribute().value((int) res));
        }
    }

    private static void TreeClassifierBuild(Instances trainSet, Instances testSet) throws Exception {
        System.out.print("-----------------------------TREE-CLASSIFIER-------------------------------\n");

        Evaluation Test = new Evaluation(trainSet);
        J48 classifier = new J48();
        classifier.setOptions(Utils.splitOptions("-C 0.1"));
        classifier.buildClassifier(trainSet);
        System.out.print(classifier.toString());

        Test.evaluateModel(classifier, testSet);
        System.out.println(Test.toSummaryString());

        System.out.print("*******************************************************************\n");
        for (int j = 0; j < testSet.numInstances(); ++j) {
            double res = classifier.classifyInstance(testSet.get(j));
            //System.out.print(testSet.classAttribute().value((int) res) + "\n");
            //System.out.print("ID: " + testSet.instance(j).value(0));
            System.out.print("Actual: " + testSet.classAttribute().value((int) testSet.instance(j).classValue()));
            System.out.println(", predicted: " + testSet.classAttribute().value((int) res));
        }
        visualizeTree(classifier);
    }

    private static void NaiveBayesClassifierBuild(Instances trainSet, Instances testSet) throws Exception {
        System.out.print("-------------------------NAIVE-BAYES-CLASSIFIER----------------------------\n");

        Evaluation eval = new Evaluation(trainSet);
        NaiveBayes classifier = new NaiveBayes();
        classifier.buildClassifier(trainSet);
        System.out.print(classifier.toString());

        eval.evaluateModel(classifier, testSet);
        System.out.println(eval.toSummaryString());

        System.out.print("*******************************************************************\n");
        for (int j = 0; j < testSet.numInstances(); ++j) {
            double res = classifier.classifyInstance(testSet.get(j));
            //System.out.print(testSet.classAttribute().value((int) res) + "\n");
            System.out.print("Actual: " + testSet.classAttribute().value((int) testSet.instance(j).classValue()));
            System.out.println(", predicted: " + testSet.classAttribute().value((int) res));
        }


        int k = 0;
        for (Instance instance : testSet) {
            double actual = instance.classValue();
            double prediction = eval.evaluateModelOnce(classifier, instance);
            System.out.printf("%2d.%4.0f%4.0f", ++k, actual, prediction);
            System.out.println(prediction != actual? " *": "");
        }
    }

    public static void visualizeTree(J48 classifier) throws Exception {
        final JFrame jf = new JFrame();
        jf.setSize(500,400);
        jf.getContentPane().setLayout(new BorderLayout());
        TreeVisualizer tv = new TreeVisualizer(null,
                classifier.graph(),
                new PlaceNode2());
        jf.getContentPane().add(tv, BorderLayout.CENTER);
        jf.addWindowListener(new java.awt.event.WindowAdapter() {
            public void windowClosing(java.awt.event.WindowEvent e) {
                jf.dispose();
            }
        });

        jf.setVisible(true);
        tv.fitToScreen();

    }
}
