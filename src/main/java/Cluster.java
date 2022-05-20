import weka.clusterers.*;
import weka.core.Instances;
import weka.core.SelectedTag;
import weka.core.Utils;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;

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

        //EMClusterBuild();
        //kMeansClusterBuild();
        //CobwebClusterBuild();
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

    public static void getEvaluation(Clusterer cluster) throws Exception {
        ClusterEvaluation eval = new ClusterEvaluation();
        eval.setClusterer(cluster);                                   // the cluster to evaluate
        eval.evaluateClusterer(data);                                // data to evaluate the clusterer on
        System.out.println("# of clusters: " + eval.getNumClusters());  // output # of clusters
        System.out.println(eval.clusterResultsToString());
    }

    public static void EMClusterBuild() throws Exception {
        System.out.println("-------------------------EM-cluster-----------------------\n");

        EM cluster = new EM();
        cluster.setOptions(Utils.splitOptions("-I 100 -N 7"));   // set the options
        cluster.buildClusterer(dataClusterer);

        getEvaluation(cluster);
    }

    public static void kMeansClusterBuild() throws Exception {
        System.out.println("-------------------------KMeans-cluster-----------------------\n");

        SimpleKMeans cluster = new SimpleKMeans();
        cluster.setOptions(Utils.splitOptions("-init 0 -I 1000 -N 7"));   // set the options
        cluster.buildClusterer(dataClusterer);

        getEvaluation(cluster);
    }

    public static void CobwebClusterBuild() throws Exception {
        System.out.println("-------------------------Cobweb-cluster-----------------------\n");

        Cobweb cluster = new Cobweb();
        cluster.setOptions(Utils.splitOptions(""));   // set the options
        cluster.buildClusterer(dataClusterer);

       getEvaluation(cluster);
    }

    public static void HierarchicalClusterBuild() throws Exception {
        System.out.println("-------------------------Hierarchical-Cluster-----------------------\n");

        HierarchicalClusterer hc = new HierarchicalClusterer();
        hc.setLinkType(new SelectedTag(4, TAGS_LINK_TYPE));
        hc.setNumClusters(7);
        hc.buildClusterer(dataClusterer);

        getEvaluation(hc);
    }
}
