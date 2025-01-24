package src.Main;

import java.io.IOException;
import java.util.ArrayList;
import src.MnistReader.MnistDataReader;
import src.MnistReader.MnistMatrix;
import src.Network.Network;

public class Main {

    public static void main(String[] args) {
        ArrayList<Integer> l = new ArrayList<>();
        l.add(2);
        l.add(2);
        l.add(1);

        Network n = new Network(l);

        MnistDataReader r = new MnistDataReader();

        MnistMatrix[] mat;

        try {
            mat = r.readData("C:\\Users\\Owner\\Documents\\neural net\\data\\train-images (1).idx3-ubyte", "C:\\Users\\Owner\\Documents\\neural net\\data\\train-labels.idx1-ubyte");
        } catch (IOException ex) {
            mat = null;
        }

    }
}
