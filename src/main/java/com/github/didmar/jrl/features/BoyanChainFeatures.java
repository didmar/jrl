package com.github.didmar.jrl.features;

import com.github.didmar.jrl.environment.discrete.BoyanChain;
import com.github.didmar.jrl.features.Features;

/**
 * Features for the {@link jrl_testing.environment.discrete.BoyanChain} environment.
 *
 * @author Didier Marin
 */
public final class BoyanChainFeatures extends Features {

	public static final double[][] phiTable = {
		{ 0.0,  0.0,  0.0,  0.0},
		{ 0.0,  0.0,  0.0,  1.0},
		{ 0.0,  0.0, 0.25, 0.75},
        { 0.0,  0.0,  0.5,  0.5},
        { 0.0,  0.0, 0.75, 0.25},
        { 0.0,  0.0,  1.0,  0.0},
        { 0.0, 0.25, 0.75,  0.0},
        { 0.0,  0.5,  0.5,  0.0},
        { 0.0, 0.75, 0.25,  0.0},
        { 0.0,  1.0,  0.0,  0.0},
        {0.25, 0.75,  0.0,  0.0},
        { 0.5,  0.5,  0.0,  0.0},
        {0.75, 0.25,  0.0,  0.0},
        { 1.0,  0.0,  0.0,  0.0}
	};

	/**
	 * Construct BoyanChainFeatures.
	 */
	public BoyanChainFeatures() {
		super(1, 4);
		assert(phiTable.length == BoyanChain.CHAIN_LENGTH);
	}

	/* (non-Javadoc)
	 * @see jrl.features.Features#phi(double[], double[])
	 */
	@Override
	public final void phi(final double[] x, final double[] y) {
		if(x.length != inDim ){
			throw new IllegalArgumentException("x must have length inDim");
		}
		if(y.length != outDim ){
			throw new IllegalArgumentException("y must have length outDim");
		}

		System.arraycopy(phiTable[(int) x[0]], 0, y, 0, 4);
	}

	@Override
	public final boolean isNormalized() {
		return false;
	}

}
