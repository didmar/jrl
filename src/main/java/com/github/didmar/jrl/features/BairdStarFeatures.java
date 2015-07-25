package com.github.didmar.jrl.features;

import org.eclipse.jdt.annotation.NonNull;

/**
 * Features for the {@link BairdStar} problem.
 * 
 * @author Didier Marin
 */
public final class BairdStarFeatures extends Features {
	
	private static final double[][] table =
		{{1,2,0,0,0,0,0},
		 {1,0,2,0,0,0,0},
		 {1,0,0,2,0,0,0},
		 {1,0,0,0,2,0,0},
		 {1,0,0,0,0,2,0},
		 {2,0,0,0,0,0,1}};
	
	public BairdStarFeatures() {
		super(1,7);
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
		
		final int s = (int) x[0];
		if(s < 0 || s >= table.length) {
			throw new IllegalArgumentException("Invalid state "+s);
		}
		
		System.arraycopy(table[s],0,y,0,outDim);
	}

	/* (non-Javadoc)
	 * @see jrl.features.Features#isNormalized()
	 */
	@Override
	public final boolean isNormalized() {
		return false;
	}
}
