package com.WCBinc.JavaNetwork.Network;

import java.util.Arrays;
import java.util.Random;

import org.ejml.data.DMatrixRMaj;
import org.ejml.dense.row.CommonOps_DDRM;

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
            DMatrixRMaj tempRandom = new DMatrixRMaj(sizes[i], 1);

            for (int j = 0; j < sizes[i]; j++) {
                tempRandom.set(j, 0, random.nextGaussian());
            }
            biases[i - 1] = tempRandom;
        }

        for (int i = 0; i < this.numLayers - 1; i++) {
            DMatrixRMaj tempRandom = new DMatrixRMaj(sizes[i + 1], sizes[i]);
            double scale = 1.0 / Math.sqrt(sizes[i]);

            for (int j = 0; j < sizes[i + 1]; j++) {
                for (int k = 0; k < sizes[i]; k++) {
                    tempRandom.set(j, k, random.nextGaussian() * scale);
                }

            }
            this.weights[i] = tempRandom;
        }

    }

    public DMatrixRMaj[] getWeights() {
        return weights;
    }

    public DMatrixRMaj[] getBiases() {
        return biases;
    }

    public void SGD(int regularizationType, double lambda, double eta, int miniBatchSize, int epochs, DMatrixRMaj[][] trainingData, DMatrixRMaj[][] testData) {
        for (int i = 0; i < epochs; i++) {
            System.out.println("Epoch: " + (i + 1));

            shuffle(trainingData);

            DMatrixRMaj[][][] miniBatches = turnIntoMinibatches(trainingData, miniBatchSize);

            //displayImage(miniBatches[0][0][0], 28, 28);
            int c = 1;
            for (DMatrixRMaj[][] miniBatch : miniBatches) {
                updateMinibatch(regularizationType, lambda, eta, miniBatch, trainingData.length);
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

    public void SGD(int regularizationType, double lambda, double eta, int miniBatchSize, int epochs, DMatrixRMaj[][] trainingData) {
        for (int i = 0; i < epochs; i++) {
            System.out.println("Epoch: " + (i + 1));

            shuffle(trainingData);

            DMatrixRMaj[][][] miniBatches = turnIntoMinibatches(trainingData, miniBatchSize);

            for (DMatrixRMaj[][] miniBatch : miniBatches) {
                updateMinibatch(regularizationType, lambda, eta, miniBatch, trainingData.length);
            }
        }
    }

    private void updateMinibatch(int regularizationType, double lambda, double eta, DMatrixRMaj[][] miniBatch, int n) {

        DMatrixRMaj[] matrixMinibatch = matrixizeMinibatch(miniBatch);

        DMatrixRMaj[][] deltas = backprop(matrixMinibatch[0], matrixMinibatch[1]);

        for (int i = 0; i < numLayers - 1; i++) {
            CommonOps_DDRM.scale(eta/ (double) miniBatch.length, deltas[0][i]);
            CommonOps_DDRM.scale(eta/ (double) miniBatch.length, deltas[1][i]);
            CommonOps_DDRM.subtract(biases[i], deltas[1][i], biases[i]);
        }
        //L2 regularization
        if (regularizationType == 0) {
            double decayVal = 1 - ((eta * lambda) / n);

            for (int i = 0; i < numLayers - 1; i++) {
                //L2 regularization
                CommonOps_DDRM.scale(decayVal, weights[i]);

                CommonOps_DDRM.subtract(weights[i], deltas[0][i], weights[i]);
            }
        }
        //L1 regularization
        else {
            double decayVal = (eta * lambda) / n;

            for (int i = 0; i < numLayers - 1; i++) {
                //L1 regularization
                DMatrixRMaj weightsSign = elementwiseSign(weights[i]);
                CommonOps_DDRM.scale(decayVal, weightsSign);
                CommonOps_DDRM.subtract(weights[i], weightsSign, weights[i]);

                CommonOps_DDRM.subtract(weights[i], deltas[0][i], weights[i]);
            }
        }
    }

    private DMatrixRMaj[][] backprop(DMatrixRMaj input, DMatrixRMaj y) {

        DMatrixRMaj[] layerError = new DMatrixRMaj[numLayers - 1];
        DMatrixRMaj[] errorW = new DMatrixRMaj[numLayers - 1];
        DMatrixRMaj[] errorB = new DMatrixRMaj[numLayers - 1];
        for (int i = 0; i < numLayers - 1; i++) {
            layerError[i] = new DMatrixRMaj(sizes[i], input.getNumCols());
            errorW[i] = new DMatrixRMaj(weights[i].getNumRows(), weights[i].getNumCols());
            errorB[i] = new DMatrixRMaj(biases[i].getNumRows(), 1);
        }

        DMatrixRMaj[] activations = new DMatrixRMaj[numLayers];

        activations[0] = input;

        DMatrixRMaj[] zs = new DMatrixRMaj[numLayers - 1];

        //forward pass
        for (int i = 1; i < numLayers; i++) {
            zs[i - 1] = feedForwardOneLayerWithoutSigmoidMatrix(activations[i - 1], i - 1);
            activations[i] = elementwiseSigmoidMatrix(zs[i - 1]);
        }

        CommonOps_DDRM.subtract(activations[numLayers - 1], y, layerError[numLayers - 2]);

        CommonOps_DDRM.multTransB(layerError[numLayers - 2], activations[numLayers - 2],  errorW[numLayers - 2]);

        errorB[numLayers - 2] = sumAllCols(layerError[numLayers - 2]);

        //backward pass
        for (int i = numLayers - 1; i > 1; i--) {
            DMatrixRMaj t3 = new DMatrixRMaj();

            CommonOps_DDRM.multTransA(weights[i - 1], layerError[i - 1], t3);

            layerError[i - 2] = hadamardMatrix(t3, elementwiseSigmoidPrimeMatrix(zs[i - 2]));

            errorB[i - 2] = sumAllCols(layerError[i - 2]);

            CommonOps_DDRM.multTransB(layerError[i - 2], activations[i - 2], errorW[i - 2]);
        }
        return new DMatrixRMaj[][]{errorW, errorB};
    }

    /*private double cross(double) {

    }*/

    private double sigmoid(double n) {
        return 1 / (1 + Math.exp(-n));
        //return Math.max(n, 0);
    }

    private double sigmoidPrime(double n) {
        double sig = sigmoid(n);
        return sig*(1-sig);
        //return n > 0 ? 1 : 0;
    }

    private DMatrixRMaj elementwiseSigmoidMatrix(DMatrixRMaj mat) {
        DMatrixRMaj out = new DMatrixRMaj(mat.getNumRows(), mat.getNumCols());
        int length = mat.data.length;

        for (int i = 0; i < length; i++) {
            out.data[i] = sigmoid(mat.data[i]);
        }
        return out;
    }

    private DMatrixRMaj elementwiseSigmoidPrimeMatrix(DMatrixRMaj mat) {
        DMatrixRMaj out = new DMatrixRMaj(mat.getNumRows(), mat.getNumCols());
        int length = mat.data.length;

        for (int i = 0; i < length; i++) {
            out.data[i] = sigmoidPrime(mat.data[i]);
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

    private DMatrixRMaj hadamardMatrix(DMatrixRMaj a, DMatrixRMaj b) {
        DMatrixRMaj out = new DMatrixRMaj(a.getNumRows(), a.getNumCols());
        int length = a.data.length;

        for (int i = 0; i < length; i++) {
            out.data[i] = a.data[i] * b.data[i];
        }

        return out;
    }

    public DMatrixRMaj feedForward(DMatrixRMaj input) {
        DMatrixRMaj tempList = new DMatrixRMaj(input);

        for (int i = 0; i < numLayers - 1; i++) {
            DMatrixRMaj tempList2 = new DMatrixRMaj();

            CommonOps_DDRM.mult(weights[i], tempList, tempList2);

            CommonOps_DDRM.add(tempList2, biases[i], tempList2);

            tempList = elementwiseSigmoidMatrix(tempList2);
        }

        return tempList;
    }

    private DMatrixRMaj feedForwardOneLayerWithoutSigmoidMatrix(DMatrixRMaj input, int layer) {
        DMatrixRMaj tempList = new DMatrixRMaj();

        CommonOps_DDRM.mult(weights[layer], input, tempList);

        addVectorToAllCols(tempList, biases[layer]);

        return tempList;
    }

    private int max(DMatrixRMaj mat) {
        int index = 0;
        double max = mat.data[0];

        int length = mat.data.length;
        for (int i = 0; i < length; i++) {
            if (mat.data[i] > max) {
                max = mat.data[i];
                index = i;
            }
        }

        return index;
    }

    private DMatrixRMaj vectArrToMatrix(DMatrixRMaj[] mat) {
        DMatrixRMaj out = new DMatrixRMaj(mat[0].getNumRows(), mat.length);


        for (int i = 0; i < mat.length; i++) {
            for (int j = 0; j < mat[0].getNumRows(); j++) {
                out.set(j, i, mat[i].get(j, 0));
            }
        }

        return out;
    }

    private void addVectorToCol(DMatrixRMaj mat, DMatrixRMaj vector, int col) {
        for (int i = 0; i < vector.getNumRows(); i++) {
            mat.set(i, col, mat.get(i, col) + vector.get(i, 0));
        }
    }

    private void addVectorToAllCols(DMatrixRMaj mat, DMatrixRMaj vector) {
        for (int i = 0; i < mat.getNumCols(); i++) {
            addVectorToCol(mat, vector, i);
        }
    }

    private DMatrixRMaj sumAllCols(DMatrixRMaj mat) {
        DMatrixRMaj out = new DMatrixRMaj(mat.getNumRows(), 1);
        for (int i = 0; i < mat.getNumRows(); i++) {
            double total = 0;
            for (int j = 0; j < mat.getNumCols(); j++) {
                total += mat.get(i, j);
            }
            out.set(i, 0, total);
        }
        return out;
    }

    private double sign(double n) {
        if (n > 0) return 1;
        return n < 0 ? -1 : 0;
    }

    private DMatrixRMaj elementwiseSign(DMatrixRMaj mat) {
        DMatrixRMaj out = new DMatrixRMaj(mat.getNumRows(), mat.getNumCols());
        int length = mat.data.length;

        for (int i = 0; i < length; i++) {
            out.data[i] = sign(mat.data[i]);
        }

        return out;
    }

    private DMatrixRMaj[] matrixizeMinibatch(DMatrixRMaj[][] mat) {
        DMatrixRMaj[] inputs = new DMatrixRMaj[mat.length];
        DMatrixRMaj[] outputs = new DMatrixRMaj[mat.length];

        for (int i = 0; i < mat.length; i++) {
            inputs[i] = mat[i][0];
            outputs[i] = mat[i][1];
        }
        DMatrixRMaj[] out = new DMatrixRMaj[2];

        out[1] = vectArrToMatrix(outputs);
        out[0] = vectArrToMatrix(inputs);

        return out;
    }

    private void shuffle(DMatrixRMaj[][] mat) {
        for (int i = mat.length - 1; i > 0; i--) {
            int r = random.nextInt(i + 1);

            DMatrixRMaj[] t = mat[i];
            mat[i] = mat[r];
            mat[r] = t;
        }
    }

    private DMatrixRMaj[][][] turnIntoMinibatches(DMatrixRMaj[][] mat, int minibatchSize) {
        int numBatches = (mat.length + minibatchSize - 1) / minibatchSize;
        DMatrixRMaj[][][] minibatches = new DMatrixRMaj[numBatches][][];

        for (int i = 0; i < numBatches; i++) {
            int start = i * minibatchSize;
            int end = Math.min(start + minibatchSize, mat.length);
            minibatches[i] = Arrays.copyOfRange(mat, start, end);
        }
        return minibatches;
    }

    public void displayImage(DMatrixRMaj mat, int height, int width) {
        int c = 0;
        for (int i = 0; i < width; i++) {
            for (int j = 0; j < height; j++) {
                if (mat.get(c,0) > 0) {
                    System.out.print("#");
                } else {
                    System.out.print(" ");
                }
                c++;
            }
            System.out.println();
        }
    }
}