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
    private static Instances data;
    private static Instances dataClusterer;

    public static void main(String[] args) throws Exception {
        data = loadDataSet("set.csv");
        weka.filters.unsupervised.attribute.Remove filter = new weka.filters.unsupervised.attribute.Remove();
        filter.setAttributeIndices("" + (data.classIndex() + 1));
        filter.setInputFormat(data);
        dataClusterer = Filter.useFilter(data, filter);

        EMClusterBuild();
        kMeansClusterBuild();
        CobwebClusterBuild();
        HierarchicalClusterBuild();
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

    public static void EMClusterBuild() throws Exception {
        System.out.println("-------------------------EM-cluster-----------------------\n");

        EM cluster = new EM();
        cluster.setOptions(weka.core.Utils.splitOptions("-I 100"));   // set the options
        cluster.setNumClusters(9);
        cluster.buildClusterer(dataClusterer);

        ClusterEvaluation eval = new ClusterEvaluation();
        eval.setClusterer(cluster);                                   // the cluster to evaluate
        eval.evaluateClusterer(data);                                // data to evaluate the clusterer on
        System.out.println("# of clusters: " + eval.getNumClusters());  // output # of clusters
        System.out.println(eval.clusterResultsToString());
    }

    public static void kMeansClusterBuild() throws Exception {
        System.out.println("-------------------------KMeans-cluster-----------------------\n");

        SimpleKMeans cluster = new SimpleKMeans();
        cluster.setNumClusters(9);
        cluster.buildClusterer(dataClusterer);

        ClusterEvaluation eval = new ClusterEvaluation();
        eval.setClusterer(cluster);                                   // the cluster to evaluate
        eval.evaluateClusterer(data);                                // data to evaluate the clusterer on
        System.out.println("# of clusters: " + eval.getNumClusters());  // output # of clusters
        System.out.println(eval.clusterResultsToString());
    }

    public static void CobwebClusterBuild() throws Exception {
        System.out.println("-------------------------Cobweb-cluster-----------------------\n");

        Cobweb cluster = new Cobweb();
        cluster.buildClusterer(dataClusterer);

        ClusterEvaluation eval = new ClusterEvaluation();
        eval.setClusterer(cluster);                                   // the cluster to evaluate
        eval.evaluateClusterer(data);                                // data to evaluate the clusterer on
        System.out.println("# of clusters: " + eval.getNumClusters());  // output # of clusters
        System.out.println(eval.clusterResultsToString());
    }
    public static void HierarchicalClusterBuild() throws Exception {
        System.out.println("-------------------------Hierarchical-Cluster-----------------------\n");

        HierarchicalClusterer hc = new HierarchicalClusterer();
        hc.setLinkType(new SelectedTag(4, TAGS_LINK_TYPE));
        hc.buildClusterer(dataClusterer);

        ClusterEvaluation eval = new ClusterEvaluation();
        eval.setClusterer(hc);                                   // the cluster to evaluate
        eval.evaluateClusterer(data);                                // data to evaluate the clusterer on
        System.out.println("# of clusters: " + eval.getNumClusters());  // output # of clusters
        System.out.println(eval.clusterResultsToString());
    }
}
