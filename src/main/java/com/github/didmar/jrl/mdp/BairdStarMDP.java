package com.github.didmar.jrl.mdp;

import com.github.didmar.jrl.evaluation.valuefunction.LinearVFunction;
import com.github.didmar.jrl.utils.array.ArrUtils;

/**
 * Baird's off-policy counterexample, as in Baird "Residual Algorithms :
 * Reinforcement Learning with Function Approximation". Uncontrolled (use a
 * dummy policy) and deterministic MDP. The reward is always zero. Use a
 * {@link LinearVFunction} with {@link BairdStarFeatures} and initial parameters
 * all positive and not all equal. With these settings, the parameters should
 * diverge for on-policy methods such as TD.
 * @author Didier Marin
 */
public final class BairdStarMDP extends DiscreteMDP {

	public BairdStarMDP() {
		super(6,1);
		ArrUtils.constvec(P0, 1./((double)n));
		for (int x = 0; x < n; x++) {
			for (int u = 0; u < m; u++) {
				ArrUtils.zeros(P[x][u]);
				P[x][u][5] = 1.;
			}
		}
		ArrUtils.zeros(R);
	}

}
