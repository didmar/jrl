package com.github.didmar.jrl.features;

import com.github.didmar.jrl.features.Features;
import com.github.didmar.jrl.mdp.DiscreteMDP;
import com.github.didmar.jrl.utils.Utils;
import com.github.didmar.jrl.utils.array.ArrUtils;

/**
 * State-action features for discrete environments.
 * @author Didier Marin
 */
public final class TabularStateActionFeatures extends Features {

	/** State-space cardinality */
	private final int n;
	
	public TabularStateActionFeatures(DiscreteMDP mdp) {
		super(2, mdp.n*mdp.m);
		this.n = mdp.n;
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
		
		//TODO add safety checks
		final int i = (int)(x[0]+n*x[1]);
		ArrUtils.zeros(y);
		y[i] = 1.;
		
		assert Utils.allClose(ArrUtils.sum(y),1.,Utils.getMacheps())
				: "Features are not normalized";
	}

	@Override
	public final boolean isNormalized() {
		return true;
	}

}
