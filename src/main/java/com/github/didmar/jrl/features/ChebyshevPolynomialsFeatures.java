package com.github.didmar.jrl.features;

import org.eclipse.jdt.annotation.NonNull;

import com.github.didmar.jrl.utils.array.ArrUtils;

/**
 * Features based on Chebyshev polynomials.
 * 
 * @author Didier Marin
 */
public final class ChebyshevPolynomialsFeatures extends Features {
	
	/** Highest degree of the polynomial */
	private final int degree;
    private final double[] scalingFactor;
    private final double[] offset;
	
	/**
	 * @param inDim
	 * @param outDim
	 * @throws IllegalArgumentException
	 */
	public ChebyshevPolynomialsFeatures(int inDim, int degree,
			final double[] inMin, final double[] inMax) {
		super(inDim, 1+degree*inDim);
		if(inDim != inMin.length || inDim != inMax.length) {
			throw new IllegalArgumentException("inMin and inMax must have length inDim");
		}
		if(!ArrUtils.allLess(inMin, inMax)) {
			throw new IllegalArgumentException("inMax must be greater or equal to inMin");
		}
		if(degree < 2) {
			throw new IllegalArgumentException("degree must be greater or equal to 2");
	    }
		this.degree = degree;
		scalingFactor = computeScalingFactor(inMin, inMax);
	    offset = computeOffset(inMin,inMax);
	}
	
	private final double[] computeScalingFactor(final double[] inMin, final double[] inMax) {
		@NonNull final double[] x = new double[inDim];
		for (int i = 0; i < inDim; i++) {
			x[i] = 2./(inMax[i]-inMin[i]);
		}
		return x;
	}
	
	private final double[] computeOffset(final double[] inMin, final double[] inMax) {
		@NonNull final double[] x = new double[inDim];
		for (int i = 0; i < inDim; i++) {
			x[i] = inMin[i] + (inMax[i]-inMin[i])/2.;
		}
		return x;
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
		
		y[0] = 1;
	    for(int k=0; k<inDim; k++)
	    {
	    	final double normX = (x[k] - offset[k]) * scalingFactor[k];
	        y[1+k*degree+0] = normX;
	        y[1+k*degree+1] = 2. * normX * y[1+k*degree+0] - 1.;
	        for(int i=2; i<degree; i++)
	        {
	            y[1+k*degree+i] = 2 * normX * y[1+k*degree+i-1] - y[1+k*degree+i-2];
	        }
	    }
	    for(int k=0; k<inDim; k++) {
	    	y[k] = 0.5*(y[k]+1.);
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
