package com.github.didmar.jrl.policy;

import org.eclipse.jdt.annotation.NonNull;

import com.github.didmar.jrl.utils.RandUtils;
import com.github.didmar.jrl.utils.array.ArrUtils;

/**
 * Log-differentiable policy for the TwoState problem proposed in Bagnell &
 * Schneider 2003 "Covariant Policy Search".
 * @author Didier Marin
 */
public final class BagnellTwoStatePolicy extends LogDifferentiablePolicy {

	private final double l;
	
	// for temporary storage to avoid mem. alloc.
	private final double[] der;
	private double e;

	public BagnellTwoStatePolicy(double l) {
		super(2);
		this.l = l;
		der = new double[2];
	}
	
	/* (non-Javadoc)
	 * @see jrl.policy.ILogDifferentiablePolicy#dLogdTheta(double[], double[])
	 */
	@NonNull
	public final double[] dLogdTheta(@NonNull double[] x,
									 @NonNull double[] u) {
		if(((int)u[0])==0) {
			//der[0] = ((int)x[0])==0 ? -l*e/(1+e) : 0;
			//der[1] = ((int)x[0])==1 ? -e/(1+e) : 0;
			der[0] = ((int)x[0])==0 ? -l*e : 0;
			der[1] = ((int)x[0])==1 ? -e   : 0;
		} else {
			der[0] = ((int)x[0])==0 ? l/(1+e) : 0;
			der[1] = ((int)x[0])==1 ? 1/(1+e) : 0;
		}
		return der;
	}

	/* (non-Javadoc)
	 * @see jrl.policy.Policy#computePolicyDistribution(double[])
	 */
	public final void computePolicyDistribution(@NonNull double[] x) {
		e = Math.exp(((int)x[0])==0 ? l*theta[0] : theta[1]);
	}

	/* (non-Javadoc)
	 * @see jrl.policy.Policy#drawAction()
	 */
	@NonNull
	public final double[] drawAction() {
		return RandUtils.nextDouble() < 1/(1+e) ?
				new double[]{0.} : new double[]{1.};
	}
	
	@SuppressWarnings("null")
	public final double[][] getProbaTable(double[][] xs) {
		final double[][] probas = new double[2][2];
		for(int x=0; x<2; x++) {
			computePolicyDistribution(xs[x]);
			probas[x][0] = 1/(1+e);
			probas[x][1] = 1 - probas[x][0];
		}
		return probas;
	}

	/* (non-Javadoc)
	 * @see jrl.policy.LogDifferentiablePolicy#boundParams(double[])
	 */
	@Override
	public final boolean boundParams(@NonNull double[] params) {
		return ArrUtils.boundVector(params, -5, +5);
	}

}
