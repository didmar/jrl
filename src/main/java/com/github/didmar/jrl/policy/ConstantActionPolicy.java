package com.github.didmar.jrl.policy;

import org.eclipse.jdt.annotation.NonNull;

/**
 * A policy that returns the same action for all states.
 * 
 * @author Didier Marin
 */
public final class ConstantActionPolicy implements Policy {

	private final double[] u;
	
	public ConstantActionPolicy(final double[] u) {
		this.u = u;
	}
	
	/* (non-Javadoc)
	 * @see jrl.policy.Policy#drawAction()
	 */
	public final double[] drawAction() {
		return u;
	}

	/* (non-Javadoc)
	 * @see jrl.policy.Policy#computePolicyDistribution(double[])
	 */
	public final void computePolicyDistribution(@NonNull final double[] x) {
		// Nothing to compute
	}

}
