package com.github.didmar.jrl.policy;

import com.github.didmar.jrl.policy.ParametricPolicy;
import com.github.didmar.jrl.utils.RandUtils;
import com.github.didmar.jrl.utils.array.ArrUtils;

/**
 * Parametric policy for the TwoState which has a performance function with
 * local optima. Useful for testing policy search methods. 
 * @author Didier Marin
 */
public final class ComplexTwoStatePolicy implements ParametricPolicy {

	/** Weight of action 1 in the current distribution */ 
	private double a;
	/** Weight of action 2 in the current distribution */
	private double b;
	/** the policy parameters */
	protected final double[] theta;
	
	public ComplexTwoStatePolicy() {
		theta = ArrUtils.zeros(2);
	}
	
	/* (non-Javadoc)
	 * @see jrl.policy.Policy#computePolicyDistribution(double[])
	 */
	@Override
	public final void computePolicyDistribution(double[] x) {
		//a = 0.001+Math.abs((x[0]==0 ? 0.5*Math.sin(theta[0])*Math.cos(theta[1]*0.2) : Math.sin(1.+0.5*theta[1])+Math.cos(-0.5+theta[0])));
		//b = 0.001+Math.abs(2+0.2*x[0]*Math.tanh(1.5+0.5*theta[0]));
		
		//a = 0.001+Math.abs((x[0]==0 ? 0.5*Math.sin(theta[0])*Math.cos(theta[1]*0.2) : Math.sin(1.+0.5*theta[1])+Math.cos(-0.5+theta[0])));
		//b = 0.001+Math.abs(0.1*Math.sin(0.5+5*theta[0]));
		
		a = 1. + 1.5*Math.exp(-ArrUtils.squaredNorm(new double[]{theta[0]+2., theta[1]-2.}));
		b = 1. + Math.exp(-0.5*ArrUtils.squaredNorm(new double[]{theta[0]-2., theta[1]+2.}));
	}

	/* (non-Javadoc)
	 * @see jrl.policy.Policy#drawAction()
	 */
	@Override
	public final double[] drawAction() {
		return RandUtils.nextDouble() < a/(a+b) ?
				new double[]{0.} : new double[]{1.};
	}

	public double[][] getProbaTable(double[][] xs) {
		final double[][] probas = new double[2][2];
		for(int x=0; x<2; x++) {
			computePolicyDistribution(xs[x]);
			probas[x][0] = a/(a+b);
			probas[x][1] = 1 - probas[x][0];
		}
		return probas;
	}
	
	/* (non-Javadoc)
	 * @see jrl.utils.ParametricFunction#boundParams(double[])
	 */
	@Override
	public final boolean boundParams(double[] params) {
		boolean bounded = false;
		for(int i=0; i<2; i++) {
			if(params[i] < -5.) {
				params[i] = -5.;
				bounded = true;
			} else if(params[i] > +5.) {
				params[i] = +5.;
				bounded = true;
			}
		}
		return bounded;
	}

	/* (non-Javadoc)
	 * @see jrl.utils.ParametricFunction#getParams()
	 */
	@Override
	public final double[] getParams() {
		return theta;
	}

	/* (non-Javadoc)
	 * @see jrl.utils.ParametricFunction#getParamsSize()
	 */
	@Override
	public final int getParamsSize() {
		return theta.length;
	}

	/* (non-Javadoc)
	 * @see jrl.utils.ParametricFunction#setParams(double[])
	 */
	@Override
	public final void setParams(double[] params) {
		if(theta.length != params.length) {
    		throw new IllegalArgumentException("Incorrect vector length");
    	}
		System.arraycopy(params, 0, theta, 0, theta.length);
	}

	/* (non-Javadoc)
	 * @see jrl.utils.ParametricFunction#updateParams(double[])
	 */
	@Override
	public final void updateParams(double[] delta) {
		if(theta.length != delta.length) {
    		throw new IllegalArgumentException("Incorrect vector length");
    	}
    	// assert theta.length == delta.length
    	for(int i=0; i<theta.length; i++) {
    		theta[i] = theta[i] + delta[i];
    	}
    	boundParams( theta );
	}

}
