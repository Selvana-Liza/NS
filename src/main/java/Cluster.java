import weka.classifiers.evaluation.Evaluation;
import weka.clusterers.*;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SelectedTag;
import weka.core.Utils;
import weka.core.converters.CSVLoader;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.gui.explorer.ClustererPanel;
import weka.gui.hierarchyvisualizer.HierarchyVisualizer;
import weka.gui.visualize.PlotData2D;
import weka.gui.visualize.VisualizePanel;
import weka.clusterers.*;
import weka.core.*;
import weka.core.converters.ConverterUtils.*;
import weka.gui.explorer.ClustererPanel;
import weka.gui.visualize.*;

import java.awt.*;
import java.io.*;
import java.text.*;
import java.util.*;

import javax.swing.*;

import javax.swing.*;

import java.awt.*;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.text.SimpleDateFormat;
import java.util.Date;

import static weka.clusterers.HierarchicalClusterer.TAGS_LINK_TYPE;

public class Cluster {

    public static void main(String[] args) throws Exception {
        String path = "set.csv";
        Instances data = loadDataSet(path);
        weka.filters.unsupervised.attribute.Remove filter = new weka.filters.unsupervised.attribute.Remove();
        filter.setAttributeIndices("" + (data.classIndex() + 1));
        filter.setInputFormat(data);
        Instances dataClusterer = Filter.useFilter(data, filter);

        EMClusterBuild(dataClusterer,data);
        kMeansClusterBuild(dataClusterer,data);
        CobwebClusterBuild(dataClusterer,data);
        HierarchicalClusterBuild(dataClusterer,data);



        /*for (Instance instance : dataClusterer) {
            System.out.printf("(%.0f,%.0f): %s%n",
                    instance.value(0), instance.value(1),
                    model.clusterInstance(instance));
        }*/


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

    public static void EMClusterBuild(Instances dataClusterer, Instances data) throws Exception {
        System.out.println("-------------------------EM-cluster-----------------------\n");

        EM cluster = new EM();
        cluster.setOptions(weka.core.Utils.splitOptions("-I 100"));   // set the options
        //cluster.setNumClusters(3);
        cluster.buildClusterer(dataClusterer);

        ClusterEvaluation eval = new ClusterEvaluation();
        eval.setClusterer(cluster);                                   // the cluster to evaluate
        eval.evaluateClusterer(data);                                // data to evaluate the clusterer on
        System.out.println("# of clusters: " + eval.getNumClusters());  // output # of clusters
        System.out.println(eval.clusterResultsToString());
    }

    public static void kMeansClusterBuild(Instances dataClusterer, Instances data) throws Exception {
        System.out.println("-------------------------KMeans-cluster-----------------------\n");

        SimpleKMeans model = new SimpleKMeans();
        model.buildClusterer(dataClusterer);

        ClusterEvaluation eval = new ClusterEvaluation();
        eval.setClusterer(model);                                   // the cluster to evaluate
        eval.evaluateClusterer(data);                                // data to evaluate the clusterer on
        System.out.println("# of clusters: " + eval.getNumClusters());  // output # of clusters
        System.out.println(eval.clusterResultsToString());
    }

    public static void CobwebClusterBuild(Instances dataClusterer, Instances data) throws Exception {
        System.out.println("-------------------------Cobweb-cluster-----------------------\n");

        Cobweb cw = new Cobweb();
        cw.buildClusterer(dataClusterer);

        ClusterEvaluation eval = new ClusterEvaluation();
        eval.setClusterer(cw);                                   // the cluster to evaluate
        eval.evaluateClusterer(data);                                // data to evaluate the clusterer on
        System.out.println("# of clusters: " + eval.getNumClusters());  // output # of clusters
        System.out.println(eval.clusterResultsToString());
    }
    public static void HierarchicalClusterBuild(Instances dataClusterer, Instances data) throws Exception {
        System.out.println("-------------------------Hierarchical-Cluster-----------------------\n");

        HierarchicalClusterer hc = new HierarchicalClusterer();
        hc.setLinkType(new SelectedTag(4, TAGS_LINK_TYPE));
        hc.buildClusterer(dataClusterer);

        ClusterEvaluation eval = new ClusterEvaluation();
        eval.setClusterer(hc);                                   // the cluster to evaluate
        eval.evaluateClusterer(data);                                // data to evaluate the clusterer on
        System.out.println("# of clusters: " + eval.getNumClusters());  // output # of clusters
        System.out.println(eval.clusterResultsToString());
        //displayDendrogram(hc.graph());
    }

    public static void displayDendrogram(String graph) {
        JFrame frame = new JFrame("Dendrogram");
        frame.setSize(500, 400);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        Container pane = frame.getContentPane();
        pane.setLayout(new BorderLayout());
        pane.add(new HierarchyVisualizer(graph));
        frame.setVisible(true);
    }

    /*public static void visualizeCluster(Instances train,Cluster cluster, Evaluation eval) throws Exception {
        PlotData2D predData = ClustererPanel.setUpVisualizableInstances(train, eval);
        String name = (new SimpleDateFormat("HH:mm:ss - ")).format(new Date());
        String cname = cluster.getClass().getName();
        if (cname.startsWith("weka.clusterers."))
            name += cname.substring("weka.clusterers.".length());
        else
            name += cname;

        VisualizePanel vp = new VisualizePanel();
        vp.setName(name + " (" + train.relationName() + ")");
        predData.setPlotName(name + " (" + train.relationName() + ")");
        vp.addPlot(predData);

        // display data
        // taken from: ClustererPanel.visualizeClusterAssignments(VisualizePanel)
        String plotName = vp.getName();
        final JFrame jf =
                new JFrame("Weka Clusterer Visualize: " + plotName);
        jf.setSize(500,400);
        jf.getContentPane().setLayout(new BorderLayout());
        jf.getContentPane().add(vp, BorderLayout.CENTER);
        jf.addWindowListener(new WindowAdapter() {
            public void windowClosing(WindowEvent e) {
                jf.dispose();
            }
        });
        jf.setVisible(true);
    }*/

}
