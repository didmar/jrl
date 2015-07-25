package com.github.didmar.jrl.features;

import org.eclipse.jdt.annotation.NonNull;

import com.github.didmar.jrl.utils.RandUtils;
import com.github.didmar.jrl.utils.array.ArrUtils;

// TODO not tested yet
/**
 * Random features based on Fourier transform.
 * 
 * @author Didier Marin
 */
public final class FourierRandomFeatures extends Features {

	private final double[][] A;
	private final double[] b;

	/**
	 * Constructor.
	 * @param inDim input dimension
	 * @param outDim output dimension
	 * @param sigma standard deviation for the Gaussian random weights
	 */
	public FourierRandomFeatures(int inDim, int outDim, double sigma) {
		super(inDim, outDim);
		A = new double[outDim][inDim];
		b = new double[outDim];
		for(int i=0; i<outDim; i++) {
			for (int j = 0; j < inDim; j++) {
				A[i][j] = RandUtils.nextGaussian(sigma);
			}
			b[i] = 2. * Math.PI * RandUtils.nextDouble();
		}
	}

	/* (non-Javadoc)
	 * @see jrl.features.Features#phi(double[], double[])
	 */
	@Override
	public final void phi(@NonNull final double[] x,
						  @NonNull final double[] y)
						 throws IllegalArgumentException {
		assert x != null;
		assert y != null;
		
		if(x.length != inDim ){
			throw new IllegalArgumentException("x must have length inDim");
		}
		if(y.length != outDim ){
			throw new IllegalArgumentException("y must have length outDim");
		}
		
		// y = cos(A*x+b)
		ArrUtils.multiply(A, x, y, outDim, inDim);
		for (int i = 0; i < outDim; i++) {
			y[i] = Math.cos(y[i]+b[i]);
		}
	}

	/* (non-Javadoc)
	 * @see jrl.features.Features#isNormalized()
	 */
	@Override
	public final boolean isNormalized() {
		return false;
	}

}
