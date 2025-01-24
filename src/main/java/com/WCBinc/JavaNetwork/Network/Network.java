package src.Network;

import java.util.ArrayList;
import java.util.Random;

import src.MnistReader.MnistMatrix;

public class Network {

    public int numLayers;

    public ArrayList<Integer> sizes = new ArrayList<>();

    private static Random random = new Random();

    public ArrayList<ArrayList<ArrayList<Double>>> weights = new ArrayList<>();
    public ArrayList<ArrayList<Double>> biases = new ArrayList<>();

    public Network(ArrayList<Integer> sizes) {
        this.numLayers = sizes.size();
        this.sizes = sizes;

        for (int i = 1; i < this.numLayers; i++) {
            ArrayList<Double> tempRandom = new ArrayList<>();
            for (int j = 0; j < sizes.get(i); j++) {
                tempRandom.add(random.nextGaussian());
            }
            biases.add(tempRandom);
        }

        for (int i = 0; i < this.numLayers - 1; i++) {
            ArrayList<ArrayList<Double>> tempRandom = new ArrayList<>();
            for (int j = 0; j < sizes.get(i + 1); j++) {
                ArrayList<Double> tempRandom2 = new ArrayList<>();
                for (int k = 0; k < sizes.get(i); k++) {
                    tempRandom2.add(random.nextGaussian());
                }
                tempRandom.add(tempRandom2);
            }
            weights.add(tempRandom);
        }

    }

    private double sigmoid(double n) {
        return 1 / (1 + Math.exp(-n));
    }

    private MnistMatrix[] shuffleMat(MnistMatrix[] mat) {
        Random r = new Random();

        for (int i = 0; i < mat.length * 3; i++) {
            int r1 = r.nextInt(0, mat.length);
            int r2 = r.nextInt(0, mat.length);

            MnistMatrix temp = mat[r1];

            mat[r1] = mat[r2];
            mat[r2] = temp;
        }

        return mat;
    }

    public ArrayList<Double> feedForward(ArrayList<Double> input) {
        ArrayList<Double> tempList = input;

        for (int i = 1; i < numLayers; i++) {
            ArrayList<Double> tempList2 = new ArrayList<>();
            for (int j = 0; j < sizes.get(i); j++) {

                double total = 0;
                for (int k = 0; k < sizes.get(i - 1); k++) {
                    total += tempList.get(k) * weights.get(i - 1).get(j).get(k);
                }

                total += biases.get(i - 1).get(j);

                tempList2.add(sigmoid(total));
            }
            tempList = tempList2;
        }

        return tempList;
    }

    public void SGD(double eta, int miniBatchSize, int epochs, MnistMatrix[] trainingData) {
        for (int i = 0; i < epochs; i++) {
            MnistMatrix[] shuffled = shuffleMat(trainingData);

            MnistMatrix[][] miniBatches = new MnistMatrix[miniBatchSize][trainingData.length / miniBatchSize];

            int c = 0;

            for (int j = 0; j < shuffled.length / miniBatchSize; j++) {
                MnistMatrix[] miniBatch = new MnistMatrix[miniBatchSize];
                for (int k = 0; k < miniBatchSize; k++) {
                    miniBatch[k] = shuffled[c];
                    c++;
                }
                miniBatches[j] = miniBatch;
            }

            for (MnistMatrix[] miniBatch : miniBatches) {
                updateMinibatch(eta, miniBatch);
            }
        }
    }

    private void updateMinibatch(double eta, MnistMatrix[] miniBatch) {

    }
}
