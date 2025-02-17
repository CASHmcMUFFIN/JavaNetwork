package com.WCBinc.JavaNetwork.MnistReader;

public class MnistMatrix {

    private int[][] data;

    private int nRows;
    private int nCols;

    private int label;

    public MnistMatrix(int nRows, int nCols) {
        this.nRows = nRows;
        this.nCols = nCols;

        data = new int[nRows][nCols];
    }

    public int getValue(int r, int c) {
        return data[r][c];
    }

    public void setValue(int row, int col, int value) {
        data[row][col] = value;
    }

    public int getLabel() {
        return label;
    }

    public void setLabel(int label) {
        this.label = label;
    }

    public int getNumberOfRows() {
        return nRows;
    }

    public int getNumberOfColumns() {
        return nCols;
    }

    public void print() {
        for (int i = 0; i < data.length; i++) {
            for (int j = 0; j < data[i].length; j++) {
                System.out.println(data[i][j] + ", ");
            }
            System.out.println();
        }
    }

}
