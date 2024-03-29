package se.kth.jabeja;

import javafx.util.Pair;
import org.apache.log4j.Logger;
import se.kth.jabeja.config.Config;
import se.kth.jabeja.config.NodeSelectionPolicy;
import se.kth.jabeja.io.FileIO;
import se.kth.jabeja.rand.RandNoGenerator;

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.*;

public class Jabeja {
  final static Logger logger = Logger.getLogger(Jabeja.class);
  private final Config config;
  private final HashMap<Integer/*id*/, Node/*neighbors*/> entireGraph;
  private final List<Integer> nodeIds;
  private int numberOfSwaps;
  private int round;
  private float T;
  private boolean resultFileCreated = false;
  private Version version;
  private int epsilon = 50;

  //-------------------------------------------------------------------
  public Jabeja(HashMap<Integer, Node> graph, Config config) {
    this.entireGraph = graph;
    this.nodeIds = new ArrayList(entireGraph.keySet());
    this.round = 0;
    this.numberOfSwaps = 0;
    this.config = config;
    this.T = config.getTemperature();
    this.version = Version.SAKATA;
  }


  //-------------------------------------------------------------------
  public void startJabeja() throws IOException {
    if (version == Version.SAKATA) T = 1;

    //config.setDelta(0.003f);
    //config.setAlpha(2.0f);

    for (round = 0; round < config.getRounds(); round++) {
      for (int id : entireGraph.keySet()) {
        if (round < 800) {
          sampleAndSwap(id);
          if (version == Version.RESTART && round % 100 == 0) {
            T = config.getTemperature();
          }
        }
        else if (version == Version.UNBA && round >= 800){
          Node nodep = entireGraph.get(id);
          int color = findColor(id);
          nodep.setColor(color);
        }
      }

      //one cycle for all nodes have completed.
      //reduce the temperature
      saCoolDown();
      report();
    }

    Map<Integer, Integer> balance = new HashMap<Integer, Integer>();

    for (int id : entireGraph.keySet()){
      Node nodep = entireGraph.get(id);
      int count = balance.containsKey(nodep.getColor()) ? balance.get(nodep.getColor()) : 0;
      balance.put(nodep.getColor(), count + 1);
    }

    /*
    try (PrintWriter out = new PrintWriter("balance.txt")) {
      out.println(balance);
    }
     */


  }

  /**
   * Simulated analealing cooling function
   */
  private void saCoolDown(){
    // Simulated Annealing verison proposed by
    if (version == Version.SAKATA){
      if (T > 0.001f) T = T*0.95f;
      if (T < 0.001f) T = 0.001f;
    }
    else if (version == Version.BASE || version == Version.RESTART || version == Version.UNBA){
      if (T > 1) T -= config.getDelta();
      if (T < 1) T = 1;
    }
  }

  /**
   * Sample and swap algorith at node p
   * @param nodeId
   */
  private void sampleAndSwap(int nodeId) {
    Node partner = null;
    Node nodep = entireGraph.get(nodeId);

    if (config.getNodeSelectionPolicy() == NodeSelectionPolicy.HYBRID
            || config.getNodeSelectionPolicy() == NodeSelectionPolicy.LOCAL) {
      // swap with random neighbors
      // TODO
      partner = this.findPartner(nodeId, getNeighbors(nodep));
    }

    if (config.getNodeSelectionPolicy() == NodeSelectionPolicy.HYBRID
            || config.getNodeSelectionPolicy() == NodeSelectionPolicy.RANDOM) {
      // if local policy fails then randomly sample the entire graph
      // TODO
      if(partner == null){
        partner = this.findPartner(nodeId, getSample(nodeId));
      }
    }

    // swap the colors
    // TODO
    if (partner!=null){
      int tempColor = partner.getColor();
      partner.setColor(nodep.getColor());
      nodep.setColor(tempColor);
      numberOfSwaps++;
    }
  }

  public int findColor(int nodeId){
    Node nodep = entireGraph.get(nodeId);
    double highestBenefit = 0;
    int color = nodep.getColor();

    for (int i= 0; i < config.getNumPartitions(); i++){
      double alpha = config.getAlpha();

      //compute the current number of neighbors with the same color
      int degpp = getDegree(nodep, nodep.getColor());
      double colNeigh = Math.pow(degpp,alpha);

      //compute the number of neighbors with the same color if you have swapped them
      int degpq = getDegree(nodep, i);
      double swapColNeigh = Math.pow(degpq,alpha);

      if(swapColNeigh > colNeigh && swapColNeigh > highestBenefit){
          color = i;
          highestBenefit = swapColNeigh;
      }
    }
    return color;
  }

  public Node findPartner(int nodeId, Integer[] nodes){

    Node nodep = entireGraph.get(nodeId);

    Node bestPartner = null;
    double highestBenefit = 0;
    ArrayList<Double> fitness = new ArrayList<Double>();

    // TODO
    for(int i: nodes){
      Node nodeq = entireGraph.get(i);
      double alpha = config.getAlpha();

      //compute the current number of neighbors with the same color
      int degpp = getDegree(nodep, nodep.getColor());
      int degqq = getDegree(nodeq, nodeq.getColor());
      double colNeigh = Math.pow(degpp,alpha) + Math.pow(degqq,alpha);

      //compute the number of neighbors with the same color if you have swapped them
      int degpq = getDegree(nodep, nodeq.getColor());
      int degqp = getDegree(nodeq, nodep.getColor());
      double swapColNeigh = Math.pow(degpq,alpha) + Math.pow(degqp, alpha);

      if (version == Version.BASE || version == Version.RESTART || version == Version.UNBA){
        if(swapColNeigh*this.T > colNeigh && swapColNeigh > highestBenefit){
          bestPartner = nodeq;
          highestBenefit = swapColNeigh;
        }
      }
      else if (version == Version.SAKATA){
        //double ap = Math.exp((swapColNeigh - colNeigh) / T);
        double ap = 1 / (1+ Math.exp((colNeigh - swapColNeigh) / T));
        Random r = new Random();
        if (r.nextDouble() < ap && swapColNeigh > highestBenefit){
          bestPartner = nodeq;
          highestBenefit = swapColNeigh;
        }
      }
    }

    return bestPartner;
  }

  /**
   * The the degreee on the node based on color
   * @param node
   * @param colorId
   * @return how many neighbors of the node have color == colorId
   */
  private int getDegree(Node node, int colorId){
    int degree = 0;
    for(int neighborId : node.getNeighbours()){
      Node neighbor = entireGraph.get(neighborId);
      if(neighbor.getColor() == colorId){
        degree++;
      }
    }
    return degree;
  }

  /**
   * Returns a uniformly random sample of the graph
   * @param currentNodeId
   * @return Returns a uniformly random sample of the graph
   */
  private Integer[] getSample(int currentNodeId) {
    int count = config.getUniformRandomSampleSize();
    int rndId;
    int size = entireGraph.size();
    ArrayList<Integer> rndIds = new ArrayList<Integer>();

    while (true) {
      rndId = nodeIds.get(RandNoGenerator.nextInt(size));
      if (rndId != currentNodeId && !rndIds.contains(rndId)) {
        rndIds.add(rndId);
        count--;
      }

      if (count == 0)
        break;
    }

    Integer[] ids = new Integer[rndIds.size()];
    return rndIds.toArray(ids);
  }

  /**
   * Get random neighbors. The number of random neighbors is controlled using
   * -closeByNeighbors command line argument which can be obtained from the config
   * using {@link Config#getRandomNeighborSampleSize()}
   * @param node
   * @return
   */
  private Integer[] getNeighbors(Node node) {
    ArrayList<Integer> list = node.getNeighbours();
    int count = config.getRandomNeighborSampleSize();
    int rndId;
    int index;
    int size = list.size();
    ArrayList<Integer> rndIds = new ArrayList<Integer>();

    if (size <= count)
      rndIds.addAll(list);
    else {
      while (true) {
        index = RandNoGenerator.nextInt(size);
        rndId = list.get(index);
        if (!rndIds.contains(rndId)) {
          rndIds.add(rndId);
          count--;
        }

        if (count == 0)
          break;
      }
    }

    Integer[] arr = new Integer[rndIds.size()];
    return rndIds.toArray(arr);
  }


  /**
   * Generate a report which is stored in a file in the output dir.
   *
   * @throws IOException
   */
  private void report() throws IOException {
    int grayLinks = 0;
    int migrations = 0; // number of nodes that have changed the initial color
    int size = entireGraph.size();

    for (int i : entireGraph.keySet()) {
      Node node = entireGraph.get(i);
      int nodeColor = node.getColor();
      ArrayList<Integer> nodeNeighbours = node.getNeighbours();

      if (nodeColor != node.getInitColor()) {
        migrations++;
      }

      if (nodeNeighbours != null) {
        for (int n : nodeNeighbours) {
          Node p = entireGraph.get(n);
          int pColor = p.getColor();

          if (nodeColor != pColor)
            grayLinks++;
        }
      }
    }

    int edgeCut = grayLinks / 2;

    logger.info("round: " + round +
            ", edge cut:" + edgeCut +
            ", swaps: " + numberOfSwaps +
            ", migrations: " + migrations);

    saveToFile(edgeCut, migrations);
  }

  private void saveToFile(int edgeCuts, int migrations) throws IOException {
    String delimiter = "\t\t";
    String outputFilePath;

    //output file name
    File inputFile = new File(config.getGraphFilePath());
    outputFilePath = config.getOutputDir() +
            File.separator +
            inputFile.getName() + "_" +
            "NS" + "_" + config.getNodeSelectionPolicy() + "_" +
            "GICP" + "_" + config.getGraphInitialColorPolicy() + "_" +
            "T" + "_" + config.getTemperature() + "_" +
            "D" + "_" + config.getDelta() + "_" +
            "RNSS" + "_" + config.getRandomNeighborSampleSize() + "_" +
            "URSS" + "_" + config.getUniformRandomSampleSize() + "_" +
            "A" + "_" + config.getAlpha() + "_" +
            "R" + "_" + config.getRounds() + ".txt";

    if (!resultFileCreated) {
      File outputDir = new File(config.getOutputDir());
      if (!outputDir.exists()) {
        if (!outputDir.mkdir()) {
          throw new IOException("Unable to create the output directory");
        }
      }
      // create folder and result file with header
      String header = "# Migration is number of nodes that have changed color.";
      header += "\n\nRound" + delimiter + "Edge-Cut" + delimiter + "Swaps" + delimiter + "Migrations" + delimiter + "Skipped" + "\n";
      FileIO.write(header, outputFilePath);
      resultFileCreated = true;
    }

    FileIO.append(round + delimiter + (edgeCuts) + delimiter + numberOfSwaps + delimiter + migrations + "\n", outputFilePath);
  }
}
