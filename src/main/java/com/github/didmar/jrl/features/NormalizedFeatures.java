package com.github.didmar.jrl.features;

import com.github.didmar.jrl.utils.Utils;
import com.github.didmar.jrl.utils.array.ArrUtils;

/**
 * Decorator to normalize some features.
 * 
 * @author Didier Marin
 */
public final class NormalizedFeatures extends Features {

	/** The features to normalize */
	private final Features baseFeat;
	
	public NormalizedFeatures(Features baseFeat) {
		super(baseFeat.inDim, baseFeat.outDim);
		this.baseFeat = baseFeat;
	}

	/* (non-Javadoc)
	 * @see jrl.features.Features#isNormalized()
	 */
	@Override
	public final boolean isNormalized() {
		return true;
	}

	/* (non-Javadoc)
	 * @see jrl.features.Features#phi(double[], double[])
	 */
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
		
		baseFeat.phi(x,y);
		ArrUtils.normalize(y);
		
		assert Utils.allClose(ArrUtils.sum(y),1.,Utils.getMacheps()) : "Features are not normalized";
	}
	
	

}
