package com.WCBinc.JavaNetwork.Main;

import java.io.IOException;
import java.util.Random;


import com.WCBinc.JavaNetwork.MnistReader.MnistConverter;
import com.WCBinc.JavaNetwork.MnistReader.MnistDataReader;
import com.WCBinc.JavaNetwork.MnistReader.MnistMatrix;
import com.WCBinc.JavaNetwork.Network.Network;
import org.ejml.data.DMatrixRMaj;

public class Main {

    public static void main(String[] args) {
        int[] l = {784, 30, 10};

        Network n = new Network(l);

        MnistDataReader reader = new MnistDataReader();
        MnistConverter converter = new MnistConverter();

        MnistMatrix[] mat;

        try {
            mat = reader.readData("C:\\Users\\Owner\\Documents\\github\\JavaNetwork\\JavaNetwork\\data\\train-images.idx3-ubyte", "C:\\Users\\Owner\\Documents\\github\\JavaNetwork\\JavaNetwork\\data\\train-labels.idx1-ubyte");
        } catch (IOException ex) {
            mat = null;
        }

        MnistMatrix[] test;

        try {
            test = reader.readData("C:\\Users\\Owner\\Documents\\github\\JavaNetwork\\JavaNetwork\\data\\t10k-images.idx3-ubyte", "C:\\Users\\Owner\\Documents\\github\\JavaNetwork\\JavaNetwork\\data\\t10k-labels.idx1-ubyte");
        } catch (IOException e) {
            test = null;
        }

        DMatrixRMaj[][] inputs = new DMatrixRMaj[mat.length][2];

        for (int i = 0; i < mat.length; i++) {
            inputs[i] = converter.convertToDMat(mat[i]);
        }

        DMatrixRMaj[][] testData = new DMatrixRMaj[test.length][2];

        for (int i = 0; i < test.length; i++) {
            testData[i] = converter.convertToDMat(test[i]);
        }

        System.out.println();

        n.SGD(3, 10, 300, inputs, testData);
    }
}
