package com.github.didmar.jrl.policy;

import com.github.didmar.jrl.utils.RandUtils;

/**
 * Uniformely Random Policy.
 * @author Didier Marin
 */
public final class UniRandomPolicy implements Policy {

	private final double[][] uBounds;

	public UniRandomPolicy(double[][] uBounds) {
		this.uBounds = uBounds;
	}
	
	/* (non-Javadoc)
	 * @see jrl.policy.Policy#computePolicyDistribution(double[])
	 */
	public final void computePolicyDistribution(double[] x) {
		// Nothing to compute
	}

	/* (non-Javadoc)
	 * @see jrl.policy.Policy#drawAction()
	 */
	public final double[] drawAction() {
		final double[] u = new double[uBounds.length];
		for(int i=0; i<u.length; i++) {
			u[i] = uBounds[i][0] + RandUtils.nextDouble()*(uBounds[i][1]-uBounds[i][0]);
		}
		return u;
	}

}
