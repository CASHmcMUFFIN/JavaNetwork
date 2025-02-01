package com.WCBinc.JavaNetwork.Network;

import java.util.Random;

import org.ejml.data.DMatrixRMaj;
import org.ejml.dense.row.CommonOps_DDRM;
import org.ejml.dense.row.mult.VectorVectorMult_DDRM;

public class Network {

    public final int numLayers;

    public final int[] sizes;

    private static final Random random = new Random();

    private final DMatrixRMaj[] weights;
    private final DMatrixRMaj[] biases;

    public Network(int[] sizes) {
        this.numLayers = sizes.length;
        this.sizes = sizes;

        this.weights = new DMatrixRMaj[numLayers - 1];
        this.biases = new DMatrixRMaj[numLayers - 1];

        for (int i = 1; i < this.numLayers; i++) {
            DMatrixRMaj tempRandom = new DMatrixRMaj(sizes[i],1);
            for (int j = 0; j < sizes[i]; j++) {
                tempRandom.set(j, random.nextGaussian());
            }
            biases[i - 1] = tempRandom;
        }

        for (int i = 0; i < this.numLayers - 1; i++) {
            DMatrixRMaj tempRandom = new DMatrixRMaj(sizes[i + 1], sizes[i]);
            for (int j = 0; j < sizes[i + 1]; j++) {
                for (int k = 0; k < sizes[i]; k++) {
                    tempRandom.set(j, k, random.nextGaussian());
                }

            }
            this.weights[i] = tempRandom;
        }

    }

    private double sigmoid(double n) {
        //return 1 / (1 + Math.exp(-n));
        return Math.max(n, 0);
    }

    private double sigmoidPrime(double n) {
        //return sigmoid(n)*(1-sigmoid(n));
        return n > 0 ? 1 : 0;
    }

    private DMatrixRMaj elementwiseSigmoid(DMatrixRMaj mat) {
        DMatrixRMaj out = new DMatrixRMaj(mat.getNumRows(), 1);

        for (int i = 0; i < mat.getNumElements(); i++) {
            out.set(i, 0, sigmoid(mat.get(i)));
        }
        return out;
    }

    private DMatrixRMaj elementwiseSigmoidPrime(DMatrixRMaj mat) {
        DMatrixRMaj out = new DMatrixRMaj(mat.getNumRows(), 1);

        for (int i = 0; i < mat.getNumElements(); i++) {
            out.set(i, 0, sigmoidPrime(mat.get(i)));
        }
        return out;
    }

    private DMatrixRMaj hadamard(DMatrixRMaj a, DMatrixRMaj b) {
        DMatrixRMaj mat = new DMatrixRMaj(a.getNumRows(), 1);

        for (int i = 0; i < a.getNumRows(); i++) {
            mat.set(i,0, a.get(i, 0)*b.get(i, 0));
        }

        return mat;
    }

    public DMatrixRMaj feedForward(DMatrixRMaj input) {
        DMatrixRMaj tempList = new DMatrixRMaj(input);

        for (int i = 0; i < numLayers - 1; i++) {
            DMatrixRMaj tempList2 = new DMatrixRMaj();

            CommonOps_DDRM.mult(weights[i], tempList, tempList2);

            CommonOps_DDRM.add(tempList2, biases[i], tempList2);

            for (int j = 0; j < tempList2.getNumElements(); j++) {
                tempList2.set(j, sigmoid(tempList2.get(j)));
            }

            tempList = tempList2;
        }

        return tempList;
    }

    private DMatrixRMaj feedForwardOneLayer(DMatrixRMaj input, int layer) {
        DMatrixRMaj tempList = new DMatrixRMaj();

        CommonOps_DDRM.mult(weights[layer], input, tempList);

        CommonOps_DDRM.add(tempList, biases[layer], tempList);

        for (int j = 0; j < tempList.getNumElements(); j++) {
            tempList.set(j, sigmoid(tempList.get(j)));
        }

        return tempList;
    }

    private DMatrixRMaj feedForwardOneLayerWithoutSigmoid(DMatrixRMaj input, int layer) {
        DMatrixRMaj tempList = new DMatrixRMaj();

        CommonOps_DDRM.mult(weights[layer], input, tempList);

        CommonOps_DDRM.add(tempList, biases[layer], tempList);

        return tempList;
    }

    private int max(DMatrixRMaj mat) {
        int index = 0;
        double max = 0;
        for (int i = 0; i < mat.getNumRows(); i++) {
            if (mat.get(i, 0) > max) {
                max = mat.get(i, 0);
                index = i;
            }
        }

        return index;
    }

    public DMatrixRMaj[] getWeights() {
        return weights;
    }

    public DMatrixRMaj[] getBiases() {
        return biases;
    }

    public void SGD(double eta, int miniBatchSize, int epochs, DMatrixRMaj[][] trainingData) {
        for (int i = 0; i < epochs; i++) {
            System.out.println("Epoch: " + i);

            shuffle(trainingData);

            DMatrixRMaj[][][] miniBatches = turnIntoMinibatches(trainingData, miniBatchSize);

            for (DMatrixRMaj[][] miniBatch : miniBatches) {
                updateMinibatch(eta, miniBatch);
            }
        }
    }

    public void SGD(double eta, int miniBatchSize, int epochs, DMatrixRMaj[][] trainingData, DMatrixRMaj[][] testData) {
        for (int i = 0; i < epochs; i++) {
            System.out.println("Epoch: " + i);

            shuffle(trainingData);

            DMatrixRMaj[][][] miniBatches = turnIntoMinibatches(trainingData, miniBatchSize);

            int c = 1;
            for (DMatrixRMaj[][] miniBatch : miniBatches) {
                updateMinibatch(eta, miniBatch);
                if (c % 1000 == 0) {
                    System.out.println(c + "/" + miniBatches.length);
                }

                c++;
            }

            int totalCorrect = 0;
            for (int j = 0; j < testData.length; j++) {
                if (max(testData[j][1]) == max(feedForward(testData[j][0]))) {
                    totalCorrect++;
                }
            }
            System.out.println("Percent correct: " + ((double)totalCorrect/(double)testData.length) * 100 + "%");
            System.out.println("Total correct: " + totalCorrect + "/" + testData.length);
            System.out.println();

            /*int totalCorrect2 = 0;
            for (int j = 0; j < trainingData.length; j++) {
                if (max(trainingData[j][1]) == max(feedForward(trainingData[j][0]))) {
                    totalCorrect2++;
                }
            }
            System.out.println("Percent correct: " + ((double)totalCorrect2/(double)trainingData.length) * 100 + "%");
            System.out.println("Total correct: " + totalCorrect2 + "/" + trainingData.length);
            System.out.println();*/
        }
    }

    private DMatrixRMaj[][][] turnIntoMinibatches(DMatrixRMaj[][] mat, int minibatchSize) {
        int numMinibatches = (int) Math.ceil((double) mat.length / minibatchSize);
        DMatrixRMaj[][][] minibatches = new DMatrixRMaj[numMinibatches][minibatchSize][2];

        for (int i = 0; i < numMinibatches; i++) {
            for (int j = 0; j < minibatchSize; j++) {
                int index = i * minibatchSize + j;
                minibatches[i][j] = mat[index];
            }
        }

        return minibatches;
    }

    public void shuffle(DMatrixRMaj[][] mat) {
        for (int i = 0; i < mat.length * 100; i++) {
            int r1 = random.nextInt(mat.length);
            int r2 = random.nextInt(mat.length);

            DMatrixRMaj[] t = mat[r1];

            mat[r1] = mat[r2];
            mat[r2] = t;
        }
    }

    private void updateMinibatch(double eta, DMatrixRMaj[][] miniBatch) {
        DMatrixRMaj[] nablaW = new DMatrixRMaj[numLayers - 1];
        for (int i = 0; i < weights.length; i++) {
            nablaW[i] = new DMatrixRMaj(weights[i].getNumRows(), weights[i].getNumCols());
        }

        DMatrixRMaj[] nablaB = new DMatrixRMaj[numLayers - 1];
        for (int i = 0; i < biases.length; i++) {
            nablaB[i] = new DMatrixRMaj(biases[i].getNumRows(), biases[i].getNumCols());
        }

        for (int i = 0; i < miniBatch.length; i++) {
            DMatrixRMaj[][] deltas = backprop(miniBatch[i][0], miniBatch[i][1]);

            for (int j = 0; j < nablaW.length; j++) {
                CommonOps_DDRM.add(deltas[0][j], nablaW[j], nablaW[j]);
            }

            for (int j = 0; j < nablaB.length; j++) {
                CommonOps_DDRM.add(deltas[1][j], nablaB[j], nablaB[j]);
            }
        }

        for (int i = 0; i < numLayers - 1; i++) {
            for (int j = 0; j < nablaW[i].getNumRows(); j++) {
                for (int k = 0; k < nablaW[i].getNumCols(); k++) {
                    nablaW[i].set(j, k, nablaW[i].get(j ,k) * (eta/ (double) miniBatch.length));
                }
            }
        }

        for (int i = 0; i < numLayers - 1; i++) {
            CommonOps_DDRM.subtract(weights[i], nablaW[i], weights[i]);
        }

        for (int i = 0; i < numLayers - 1; i++) {
            for (int j = 0; j < nablaB[i].getNumRows(); j++) {
                nablaB[i].set(j, 0, nablaB[i].get(j ,0) * (eta/ (double) miniBatch.length));
            }
        }

        for (int i = 0; i < numLayers - 1; i++) {
            CommonOps_DDRM.subtract(biases[i], nablaB[i], biases[i]);
        }
    }

    private DMatrixRMaj[][] backprop(DMatrixRMaj input, DMatrixRMaj y) {

        DMatrixRMaj[] layerError = new DMatrixRMaj[numLayers - 1];
        for (int i = 0; i < numLayers - 1; i++) {
            layerError[i] = new DMatrixRMaj(sizes[i], 1);
        }

        DMatrixRMaj[] errorW = new DMatrixRMaj[numLayers - 1];
        for (int i = 0; i < weights.length; i++) {
            errorW[i] = new DMatrixRMaj(weights[i].getNumRows(), weights[i].getNumCols());
        }
        DMatrixRMaj[] errorB = new DMatrixRMaj[numLayers - 1];
        for (int i = 0; i < biases.length; i++) {
            errorB[i] = new DMatrixRMaj(biases[i].getNumRows(), 1);
        }

        DMatrixRMaj[] activations = new DMatrixRMaj[numLayers];

        activations[0] = input;

        DMatrixRMaj[] zs = new DMatrixRMaj[numLayers - 1];

        for (int i = 1; i < numLayers; i++) {
            zs[i - 1] = feedForwardOneLayerWithoutSigmoid(activations[i - 1], i - 1);
            activations[i] = elementwiseSigmoid(zs[i - 1]);
        }

        DMatrixRMaj t = new DMatrixRMaj();
        layerError[numLayers - 2] = hadamard(CommonOps_DDRM.subtract(activations[numLayers - 1], y, t), elementwiseSigmoidPrime(zs[numLayers - 2]));

        VectorVectorMult_DDRM.outerProd(layerError[numLayers - 2], activations[numLayers - 2],  errorW[numLayers - 2]);

        errorB[numLayers - 2] = layerError[numLayers - 2];

        for (int i = numLayers - 1; i >= 2; i--) {
            DMatrixRMaj t2 = new DMatrixRMaj();
            DMatrixRMaj t3 = new DMatrixRMaj();

            CommonOps_DDRM.transpose(weights[i - 1], t2);
            CommonOps_DDRM.mult(t2, layerError[i - 1], t3);

            layerError[i - 2] = hadamard(t3, elementwiseSigmoidPrime(activations[i - 1]));

            errorB[i - 2] = layerError[i - 2];
            VectorVectorMult_DDRM.outerProd(layerError[i - 2], activations[i - 2], errorW[i - 2]);
        }

        DMatrixRMaj[][] val = new DMatrixRMaj[2][numLayers - 1];
        val[0] = errorW;
        val[1] = errorB;

        return val;
    }
}