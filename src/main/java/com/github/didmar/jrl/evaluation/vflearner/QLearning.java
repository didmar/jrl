package com.github.didmar.jrl.evaluation.vflearner;

import com.github.didmar.jrl.environment.EnvironmentListener;
import com.github.didmar.jrl.evaluation.valuefunction.LinearQFunction;
import com.github.didmar.jrl.evaluation.valuefunction.QFunction;
import com.github.didmar.jrl.stepsize.StepSize;
import com.github.didmar.jrl.utils.DiscountFactor;
import com.github.didmar.jrl.utils.array.ArrUtils;

/**
 * Q(lambda) implementation.
 * @author Didier Marin
 */
public final class QLearning implements QFunctionLearner, EnvironmentListener {

	/** State action value function approximator */
	private final LinearQFunction qFunction;
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
	/** Sample actions set */
	protected final double[][] actions;
	
	// arrays for temporary storage to avoid mem. alloc.
	/** Used to concatenate a state and an action */
	private final double[] xu;
	/** Used to get the state-action features of the Q-function approximator */
	private final double[] psixu;
	
	public QLearning(LinearQFunction qFunction, double[][] actions,
			DiscountFactor gamma, DiscountFactor lambda, StepSize stepSize) {
		this.qFunction = qFunction;
		this.actions = actions;
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
		// Get the best action for state x with respect to Q
		int indBest = 0;
		double QBest = qFunction.get(x, actions[0]);
		for (int i = 0; i < actions.length; i++) {
			double Q = qFunction.get(x, actions[i]);
			if(Q > QBest) {
				indBest = i;
				QBest = Q;
			}
		}
		// Does u correspond to the best action in terms of Q-value ?
		final boolean greedyAction = u.equals(actions[indBest]);
		double currentQ;
		if(greedyAction) {
			currentQ = QBest;
		} else {
			currentQ = qFunction.get(x, u);
		}
		// Compute the Q-Learning TD Error
		double tdErr = r - currentQ;
		if(!isTerminal) {
			// Get the best action for state xn with respect to Q
			double nextQBest = qFunction.get(xn, actions[0]);
			for (int i = 0; i < actions.length; i++) {
				final double Q = qFunction.get(xn, actions[i]);
				if(Q > nextQBest) {
					nextQBest = Q;
				}
			}
			tdErr += gamma.value*nextQBest;
		}
		// Compute the eligibility traces
		System.arraycopy(x, 0, xu, 0, xDim);
		System.arraycopy(u, 0, xu, xDim, uDim);
		qFunction.getFeatures().phi(xu,psixu);
		if(greedyAction) {
			for(int i=0; i<n; i++) {
				eligib[i] = gamma.value*lambda.value*eligib[i] + psixu[i];
			}
		} else {
			// If u was an exploratory action, forget the traces
			for(int i=0; i<n; i++) {
				eligib[i] = psixu[i];
			}
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
