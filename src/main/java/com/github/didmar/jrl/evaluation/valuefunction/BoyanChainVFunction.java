package com.github.didmar.jrl.evaluation.valuefunction;

import com.github.didmar.jrl.environment.discrete.BoyanChain;
import com.github.didmar.jrl.features.BoyanChainFeatures;

/**
 * State value function of the Boyan Chain environment.
 * @see BoyanChain
 * @author Didier Marin
 */
public final class BoyanChainVFunction extends LinearVFunction {

	public BoyanChainVFunction() {
		super(new BoyanChainFeatures(), new double[]{-24.0, -16.0, -8.0, 0.0});
	}

}
