package com.WCBinc.JavaNetwork.Main;

import java.io.IOException;
import java.util.ArrayList;
import com.WCBinc.JavaNetwork.MnistReader.MnistDataReader;
import com.WCBinc.JavaNetwork.MnistReader.MnistMatrix;
//import com.WCBinc.JavaNetwork.Network.Network;

public class Main {

    public static void main(String[] args) {
        ArrayList<Integer> l = new ArrayList<>();
        l.add(2);
        l.add(2);
        l.add(1);

        //Network n = new Network(l);

        MnistDataReader r = new MnistDataReader();

        MnistMatrix[] mat;

        try {
            mat = r.readData("C:\\Users\\Owner\\Documents\\github\\JavaNetwork\\JavaNetwork\\data\\train-images.idx3-ubyte", "C:\\Users\\Owner\\Documents\\github\\JavaNetwork\\JavaNetwork\\data\\train-labels.idx1-ubyte");
        } catch (IOException ex) {
            mat = null;
        }

        System.out.println(mat);

    }
}
