package com.github.didmar.jrl.agent;

import com.github.didmar.jrl.policy.Policy;

/**
 * An agent which behave according to a given policy.
 * 
 * @author Didier Marin
 */
public class PolicyAgent implements Agent {

	/** The policy it takes actions from */
	protected Policy pol;

	public PolicyAgent(Policy pol) {
		if(pol == null) {
			throw new IllegalArgumentException("pol must be non-null");
		}
		this.pol = pol;
	}

	/**
	 * Takes an action according to its policy
	 */
	public final double[] takeAction(double[] x) {
		assert x != null;
		
		pol.computePolicyDistribution(x);
	    return pol.drawAction();
	}
}
