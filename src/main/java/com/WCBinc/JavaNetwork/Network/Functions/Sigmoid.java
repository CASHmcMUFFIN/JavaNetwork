package com.WCBinc.JavaNetwork.Network.Functions;

public class Sigmoid implements NeuronFunction {

    @Override
    public double func(double n) {
        return 1 / (1 + Math.exp(-n));
    }

    @Override
    public double delta(double n) {
        double sig = func(n);
        return sig*(1-sig);
    }
}
