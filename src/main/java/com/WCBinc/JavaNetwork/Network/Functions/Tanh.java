package com.WCBinc.JavaNetwork.Network.Functions;

public class Tanh implements NeuronFunction {
    @Override
    public double func(double n) {
        return Math.tanh(n);
    }

    @Override
    public double delta(double n) {
        return 2/(Math.exp(n)+Math.exp(-n));
    }
}
