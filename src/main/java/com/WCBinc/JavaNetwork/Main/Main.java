package com.WCBinc.JavaNetwork.Main;

import java.io.IOException;
import java.util.Arrays;

import com.WCBinc.JavaNetwork.MnistReader.MnistConverter;
import com.WCBinc.JavaNetwork.MnistReader.MnistDataReader;
import com.WCBinc.JavaNetwork.MnistReader.MnistMatrix;
import com.WCBinc.JavaNetwork.Network.Network;
import org.ejml.data.DMatrixRMaj;

public class Main {
    public static void main(String[] args) {
        int[] l = {784, 100, 10};

        Network n = new Network(l);

        MnistDataReader reader = new MnistDataReader();

        MnistMatrix[] mat;

        try {
            mat = reader.readData("C:\\Users\\Owner\\Documents\\github\\JavaNetwork\\JavaNetwork\\data\\" +
                    "train-images.idx3-ubyte", "C:\\Users\\Owner\\Documents\\github\\JavaNetwork\\JavaNetwork\\data\\train-labels.idx1-ubyte");
        } catch (IOException ex) {
            mat = null;
        }

        MnistMatrix[] test;

        try {
            test = reader.readData("C:\\Users\\Owner\\Documents\\github\\JavaNetwork\\JavaNetwork\\data" +
                    "\\t10k-images.idx3-ubyte", "C:\\Users\\Owner\\Documents\\github\\JavaNetwork\\JavaNetwork\\data\\t10k-labels.idx1-ubyte");
        } catch (IOException e) {
            test = null;
        }
        DMatrixRMaj[][] inputs = new DMatrixRMaj[mat.length][2];

        for (int i = 0; i < mat.length; i++) {
            inputs[i] = MnistConverter.convertToDMat(mat[i]);
        }

        DMatrixRMaj[][] testData = new DMatrixRMaj[test.length][2];

        for (int i = 0; i < test.length; i++) {
            testData[i] = MnistConverter.convertToDMat(test[i]);
        }


        System.out.println();

        DMatrixRMaj[][] valid = Arrays.copyOfRange(inputs, 50000, 60000);
        DMatrixRMaj[][] realinp = Arrays.copyOfRange(inputs, 0, 50000);

        long time = System.currentTimeMillis();
        //n.SGD(1,0.01, 0.0015, 10, 60, realinp, valid);
        n.SGD(0,100, 0.1, 10, 60, realinp, valid);
        System.out.println(System.currentTimeMillis() - time);
    }
}