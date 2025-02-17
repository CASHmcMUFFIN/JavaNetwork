package com.WCBinc.JavaNetwork.MnistReader;

import org.ejml.data.DMatrixRMaj;

public class MnistConverter {
    private MnistConverter() {

    }

    public static DMatrixRMaj[] convertToDMat(MnistMatrix mat) {
        double[] data = new double[mat.getNumberOfRows() * mat.getNumberOfColumns()];

        DMatrixRMaj dmat = new DMatrixRMaj(mat.getNumberOfRows() * mat.getNumberOfColumns(), 1);

        int c = 0;
        for (int i = 0; i < mat.getNumberOfRows(); i++) {
            for (int j = 0; j < mat.getNumberOfColumns(); j++) {
                dmat.set(c, (double) mat.getValue(i, j) / 255.0);
                c++;
            }
        }

        DMatrixRMaj[] full = new DMatrixRMaj[2];

        full[0] = dmat;

        DMatrixRMaj label = vectorize(mat.getLabel());

        full[1] = label;

        return full;
    }

    private static DMatrixRMaj vectorize(int n) {
        DMatrixRMaj mat = new DMatrixRMaj(10, 1);

        mat.zero();

        mat.set(n, 1);

        return mat;
    }
}
