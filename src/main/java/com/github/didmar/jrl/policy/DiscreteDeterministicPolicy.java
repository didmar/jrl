package com.github.didmar.jrl.policy;

import org.eclipse.jdt.annotation.NonNull;


/**
 * A deterministic policy for discrete state-spaces.
 * @author Didier Marin
 */
public final class DiscreteDeterministicPolicy implements Policy {

	private final double[] pol;
	private final double[] u;

	public DiscreteDeterministicPolicy(final int[] pol) {
		this.pol = new double[pol.length];
		for(int i=0; i<pol.length; i++) {
			this.pol[i] = pol[i];
		}
		u = new double[1];
	}

	/* (non-Javadoc)
	 * @see jrl.policy.Policy#computePolicyDistribution(double[])
	 */
	public final void computePolicyDistribution(@NonNull final double[] x) {
		final int s = (int) x[0];
		u[0] = pol[s];
	}

	/* (non-Javadoc)
	 * @see jrl.policy.Policy#drawAction()
	 */
	@NonNull
	public final double[] drawAction() {
		return u;
	}

}
