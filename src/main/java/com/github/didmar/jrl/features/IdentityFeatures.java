package com.github.didmar.jrl.features;

import org.eclipse.jdt.annotation.NonNull;

/**
 * The identity function.
 * 
 * @author Didier Marin
 */
public class IdentityFeatures extends Features {

	public IdentityFeatures(int dim) {
		super(dim, dim);
	}

	/* (non-Javadoc)
	 * @see jrl.features.Features#phi(double[], double[])
	 */
	@Override
	public void phi(@NonNull final double[] x, @NonNull final double[] y) {
		assert x != null;
		assert y != null;
		
		if(x.length != inDim ){
			throw new IllegalArgumentException("x must have length inDim");
		}
		if(y.length != outDim ){
			throw new IllegalArgumentException("y must have length outDim");
		}
		
		System.arraycopy(x, 0, y, 0, inDim);
	}

	/* (non-Javadoc)
	 * @see jrl.features.Features#isNormalized()
	 */
	@Override
	public boolean isNormalized() {
		return false;
	}

}
