package com.WCBinc.JavaNetwork.Network;

import com.WCBinc.JavaNetwork.Network.Functions.NeuronFunction;
import org.ejml.data.DMatrixRMaj;
import org.ejml.dense.row.CommonOps_DDRM;

public class NetworkFunctions {
    public NeuronFunction function;

    public NetworkFunctions(NeuronFunction function) {
        this.function = function;
    }

    public DMatrixRMaj elementwiseFunc(DMatrixRMaj mat) {
        DMatrixRMaj out = new DMatrixRMaj(mat.getNumRows(), mat.getNumCols());
        int length = mat.data.length;

        for (int i = 0; i < length; i++) {
            out.data[i] = function.func(mat.data[i]);
        }
        return out;
    }

    public DMatrixRMaj elementwiseDelta(DMatrixRMaj mat) {
        DMatrixRMaj out = new DMatrixRMaj(mat.getNumRows(), mat.getNumCols());
        int length = mat.data.length;

        for (int i = 0; i < length; i++) {
            out.data[i] = function.delta(mat.data[i]);
        }
        return out;
    }

    public DMatrixRMaj feedForward(DMatrixRMaj input, int numlayers, DMatrixRMaj[] weights, DMatrixRMaj[] biases) {
        DMatrixRMaj tempList = new DMatrixRMaj(input);

        for (int i = 0; i < numlayers - 1; i++) {
            DMatrixRMaj tempList2 = new DMatrixRMaj();

            CommonOps_DDRM.mult(weights[i], tempList, tempList2);

            CommonOps_DDRM.add(tempList2, biases[i], tempList2);

            tempList = elementwiseFunc(tempList2);
        }

        return tempList;
    }

    public DMatrixRMaj feedForwardOneLayerWithoutFunc(DMatrixRMaj input, int layer, DMatrixRMaj[] weights, DMatrixRMaj[] biases) {
        DMatrixRMaj tempList = new DMatrixRMaj();

        CommonOps_DDRM.mult(weights[layer], input, tempList);

        addVectorToAllCols(tempList, biases[layer]);

        return tempList;
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
}
