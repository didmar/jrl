package com.github.didmar.jrl.evaluation.vflearner.td;

import com.github.didmar.jrl.evaluation.valuefunction.LinearVFunction;
import com.github.didmar.jrl.stepsize.StepSize;
import com.github.didmar.jrl.utils.DiscountFactor;
import com.github.didmar.jrl.utils.array.ArrUtils;

/**
 * Implementation of TD(lambda). For TD(0), prefer {@link TDZero}.
 * @author Didier Marin
 */
public final class TDLambda extends TDZero {

	/** Eligibility factor */
	protected final DiscountFactor lambda;
	/** Number of value function parameters */
	protected final int n;
	/** Eligibility traces */
	protected final double[] eligib;
	
	public TDLambda(LinearVFunction vFunction, StepSize stepSize,
			DiscountFactor gamma, DiscountFactor lambda) {
		super(vFunction, stepSize, gamma);
		this.lambda = lambda;
		n = vFunction.getParamsSize();
		eligib = new double[n];
	}
	
	/**
	 * Reset the eligibility traces at the beginning of each episode
	 */
	@Override
	public void newEpisode(double[] x0, int maxT) {
		ArrUtils.zeros(eligib);
	}

	/* (non-Javadoc)
	 * @see jrl.evaluation.valuefunction.TDZeroLearner#addSample(double[], double[], double, boolean)
	 */
	@Override
	public void addSample(double[] x, double[] xn, double r, boolean isTerminal) {
		// Update the step-size
		stepSize.updateStep();
		// Compute the TD Error
		final double tdErr = vFunction.tdError(x, xn, r, isTerminal, gamma);
		// Update the eligibility traces
		final double[] phix =  ((LinearVFunction)vFunction).getFeatures().phi(x);
		for(int i=0; i<n; i++) {
			eligib[i] = gamma.value*lambda.value*eligib[i] + phix[i];
		}
		((LinearVFunction)vFunction).updateForFeaturesVector(eligib, stepSize.getStep()*tdErr);
	}
	
	@Override
	public String toString() {
		return "TD("+lambda.value+")";
	}
}
