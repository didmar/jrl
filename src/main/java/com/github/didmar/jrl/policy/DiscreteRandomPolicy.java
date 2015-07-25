package com.github.didmar.jrl.policy;

import org.eclipse.jdt.annotation.NonNull;
import org.eclipse.jdt.annotation.Nullable;

import com.github.didmar.jrl.utils.RandUtils;
import com.github.didmar.jrl.utils.array.ArrUtils;

/**
 * Draw actions from a set of possible actions with equal probability,
 * ignoring the state
 * 
 * @author Didier Marin
 */
public class DiscreteRandomPolicy implements Policy {

	/** Set of possible actions with equal probability */
	final double[][] actions;
	
	/**
	 * @param actions	set of possible actions
	 */
	public DiscreteRandomPolicy(final double[][] actions) {
		if(! ArrUtils.isMatrix(actions)) {
			throw new IllegalArgumentException("actions array is not a matrix");
		}
		this.actions = actions;
	}
	
	/* (non-Javadoc)
	 * @see jrl.policy.Policy#computePolicyDistribution(double[])
	 */
	public void computePolicyDistribution(@NonNull final double[] x) {
		// Nothing to compute
	}

	/* (non-Javadoc)
	 * @see jrl.policy.Policy#drawAction()
	 */
	@NonNull
	public double[] drawAction() {
		final int i = RandUtils.nextInt(actions.length);
		@Nullable final double[] u = actions[i];
		if(u==null) throw new RuntimeException("null action within the actions set");
		return u;
	}

}
