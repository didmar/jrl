package com.github.didmar.jrl.evaluation.vflearner;

import com.github.didmar.jrl.environment.EnvironmentListener;
import com.github.didmar.jrl.evaluation.valuefunction.LinearQFunction;
import com.github.didmar.jrl.evaluation.valuefunction.QFunction;
import com.github.didmar.jrl.policy.Policy;
import com.github.didmar.jrl.stepsize.StepSize;
import com.github.didmar.jrl.utils.DiscountFactor;
import com.github.didmar.jrl.utils.array.ArrUtils;

/**
 * SARSA(lambda) implementation.
 * @author Didier Marin
 */
public final class SARSALambda implements QFunctionLearner, EnvironmentListener {

	/** State action value function approximator */
	private final LinearQFunction qFunction;
	/** Policy being evaluated */
	private final Policy pol;
	/** Reward discount factor */
	private final DiscountFactor gamma;
	/** Eligibility factor */
	private final DiscountFactor lambda;
	/** Learning step */
	private final StepSize stepSize;
	/** State-space dimension */
	private final int xDim;
	/** Action-space dimension */
	private final int uDim;
	/** Number of Q-function approximation parameters */
	private final int n;
	/** Eligibility traces */
	private final double[] eligib;
	
	// arrays for temporary storage to avoid mem. alloc.
	/** Used to concatenate a state and an action */
	private final double[] xu;
	/** Used to get the state-action features of the Q-function approximator */
	private final double[] psixu;
	
	public SARSALambda(LinearQFunction qFunction, Policy pol,
			DiscountFactor gamma, DiscountFactor lambda, StepSize stepSize) {
		this.qFunction = qFunction;
		this.pol = pol;
		this.gamma = gamma;
		this.lambda = lambda;
		this.stepSize = stepSize;
		xDim = qFunction.getXDim();
		uDim = qFunction.getUDim();
		n = qFunction.getParamsSize();
		eligib = new double[n];
		
		xu = new double[xDim+uDim];
		psixu = new double[n];
	}
	
	@Override
	public final void newEpisode(double[] x0, int maxT) {
		ArrUtils.zeros(eligib);
	}

	@Override
	public final void receiveSample(double[] x, double[] u, double[] xn, double r, boolean isTerminal) {
		// Update the step-size
		stepSize.updateStep();
		// Compute the next action from the policy
		pol.computePolicyDistribution(xn);
		final double[] un = pol.drawAction();
		// Compute the SARSA TD Error
		double tdErr = qFunction.tdError(x, u, xn, un, r, isTerminal, gamma);
		// Compute the eligibility traces
		System.arraycopy(x, 0, xu, 0, xDim);
		System.arraycopy(u, 0, xu, xDim, uDim);
		qFunction.getFeatures().phi(xu,psixu);
		for(int i=0; i<n; i++) {
			eligib[i] = gamma.value*lambda.value*eligib[i] + psixu[i];
		}
		qFunction.updateForFeaturesVector(eligib, stepSize.getStep()*tdErr);
	}

	@Override
	public final void endEpisode() {
		// Nothing to do
	}

	@Override
	public final QFunction getQFunction() {
		return qFunction;
	}

}
