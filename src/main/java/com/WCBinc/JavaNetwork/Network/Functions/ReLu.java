package com.WCBinc.JavaNetwork.Network.Functions;

public class ReLu implements NeuronFunction {

    @Override
    public double func(double n) {
        return Math.max(n, 0);
    }

    @Override
    public double delta(double n) {
        return n > 0 ? 1 : 0;
    }
}
