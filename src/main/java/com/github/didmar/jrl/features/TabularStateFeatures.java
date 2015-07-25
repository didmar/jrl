package com.github.didmar.jrl.features;

import com.github.didmar.jrl.features.Features;
import com.github.didmar.jrl.mdp.DiscreteMDP;
import com.github.didmar.jrl.utils.Utils;
import com.github.didmar.jrl.utils.array.ArrUtils;

/**
 * State features for discrete environments.
 * 
 * @author Didier Marin
 */
public final class TabularStateFeatures extends Features {

	public TabularStateFeatures(DiscreteMDP mdp) {
		super(1, mdp.n);
	}

	/* (non-Javadoc)
	 * @see jrl.features.Features#phi(double[], double[])
	 */
	@Override
	public final void phi(double[] x, double[] y) {
		if(x==null) throw new IllegalArgumentException("x must not be null");
		if(y==null) throw new IllegalArgumentException("y must not be null");
		if(x.length != inDim ){
			throw new IllegalArgumentException("x must have length inDim");
		}
		if(y.length != outDim ){
			throw new IllegalArgumentException("y must have length outDim");
		}
		
		final int i = (int)x[0];
		if(i < 0 || i >= outDim) {
			throw new IllegalArgumentException(
					"state "+i+" is not within [0,n-1[");
		}
		
		ArrUtils.zeros(y);
		y[(int)x[0]] = 1.;
		
		assert Utils.allClose(ArrUtils.sum(y),1.,Utils.getMacheps())
				: "Features are not normalized";
	}

	@Override
	public final boolean isNormalized() {
		return true;
	}

}
