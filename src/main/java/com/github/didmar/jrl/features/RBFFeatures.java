package com.github.didmar.jrl.features;

import com.github.didmar.jrl.utils.Utils;
import com.github.didmar.jrl.utils.array.ArrUtils;

/**
 * Gaussian Radial Basis Function (RBF) features.
 * 
 * @author Didier Marin
 */
public final class RBFFeatures extends Features {

	/** RBF centers (outDim-by-inDim matrix) */
	private final double[][] c;
	/** RBF standard deviation (inDim vector) */
	private final double[] sigma;
	/** Indicates whether to normalize the features or not */
	private final boolean normalized;
	
	public RBFFeatures(double[][] c, double[] sigma, boolean normalized) {
		super(c[0].length, c.length);
		if(sigma.length != inDim) {
			throw new IllegalArgumentException("sigma must have length inDim");
		}
		this.c = c;
		this.sigma = sigma;
		this.normalized = normalized; 
		// assert c[0].length == sigma.length
	}

	@Override
	public final void phi(double[] x, double[] y) {
		assert x != null;
		assert y != null;
		
		if(x.length != inDim ){
			throw new IllegalArgumentException("x must have length inDim");
		}
		if(y.length != outDim ){
			throw new IllegalArgumentException("y must have length outDim");
		}
		
		for(int i=0; i<outDim; i++) {
			double s = 0;
			for(int j=0; j<inDim; j++) {
				s += Math.pow(x[j]-c[i][j],2) / sigma[j]; 
			}
			y[i] = Math.exp(-s);
		}
		if(normalized) {
	        double sum_y = ArrUtils.sum(y);
	        for(int i=0; i<y.length; i++) {
	        	y[i] = y[i] / sum_y;
	        }
	        assert Utils.allClose(ArrUtils.sum(y),1.,10*Utils.getMacheps()) : "Features are not normalized";
		}
	}

	@Override
	public final boolean isNormalized() {
		return normalized;
	}
	
	@Override
	public final String toString() {
		return "RBFFeatures [c=" + ArrUtils.toString(c) + ", sigma="
				+ ArrUtils.toString(sigma) + ", normalized=" + normalized + "]";
	}
}
