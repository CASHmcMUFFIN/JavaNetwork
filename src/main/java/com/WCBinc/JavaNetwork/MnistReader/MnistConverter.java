package com.WCBinc.JavaNetwork.MnistReader;

import org.ejml.data.DMatrixRMaj;

public class MnistConverter {
    public DMatrixRMaj[] convertToDMat(MnistMatrix mat) {
        double[] data = new double[mat.getNumberOfRows() * mat.getNumberOfColumns()];

        DMatrixRMaj dmat = new DMatrixRMaj(mat.getNumberOfRows() * mat.getNumberOfColumns(), 1);

        int c = 0;
        for (int i = 0; i < mat.getNumberOfRows(); i++) {
            for (int j = 0; j < mat.getNumberOfColumns(); j++) {
                dmat.set(c, mat.getValue(i, j));
                c++;
            }
        }

        DMatrixRMaj[] full = new DMatrixRMaj[2];

        full[0] = dmat;

        DMatrixRMaj label = vectorize(mat.getLabel());

        full[1] = label;

        return full;
    }

    private DMatrixRMaj vectorize(int n) {
        DMatrixRMaj mat = new DMatrixRMaj(10, 1);

        mat.zero();

        mat.set(n, 1);

        return mat;
    }
}
