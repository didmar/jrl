package com.github.didmar.jrl.policy;

import java.util.Random;

import com.github.didmar.jrl.evaluation.valuefunction.QFunction;
import com.github.didmar.jrl.utils.array.ArrUtils;

/**
 * Policy that chooses the best action with respect to a given Q-function with
 * probability <code>1.-eps</code>, or a random action in a given set otherwise.
 * @author Didier Marin
 */
public final class EpsGreedyPolicyOverQ implements QFunctionBasedPolicy {

	private final static Random rand = new Random();

	/** State-action value function to compute the action distribution */
	private final QFunction qFunction;
	/** Set of possible actions */
	private final double[][] actions;
	/** Number of possible actions */
	private final int nActions;
	/** Greediness of the policy, i.e. the probability to draw a random action
	 * instead of the best in terms of Q-value. Must be within [0,1]. */
	private double eps;
	/** Action-space dimension */
	private final int uDim;

	// arrays for temporary storage to avoid mem. alloc.
	/** Used to store the Q-value for each sample action */
	private final double[] QValues;
	/** Index of the best action in the actions set for the current state */
	private int indBestAction;

	public EpsGreedyPolicyOverQ(QFunction qFunction, double[][] actions,
			double eps) {
		this.qFunction = qFunction;
		this.actions = actions;
		nActions = this.actions.length;
		setEps(eps);
		uDim = this.qFunction.getUDim();
		for(int i=0; i<nActions; i++) {
			if(uDim != this.actions[i].length) {
				throw new IllegalArgumentException("Some sample action " +
						"dimension is not compatible with the Q-function " +
						"action dimension");
			}
		}

		// arrays for temporary storage to avoid mem. alloc.
		QValues = new double[nActions];
	}

	/* (non-Javadoc)
	 * @see jrl.policy.Policy#computePolicyDistribution(double[])
	 */
	@Override
	public final void computePolicyDistribution(double[] x) {
		// Get the Q-value for each sample action
		System.out.print("x="+ArrUtils.toString(x)+" Qvalues=");
        for(int i=0; i<nActions; i++){
        	QValues[i] = qFunction.get(x, actions[i]);
        	System.out.print(QValues[i]+" ");
        }
        // Get the best action in terms of Q-value
        indBestAction = ArrUtils.argmax(QValues);
        System.out.print(" indBest="+indBestAction);
        System.out.println();
	}

	/* (non-Javadoc)
	 * @see jrl.policy.Policy#drawAction()
	 */
	@Override
	public final double[] drawAction() {
		if(Math.random() <= eps) {
			// Draw a random action
			return actions[rand.nextInt(nActions)];
		}
		// Else, take the best action
		return actions[indBestAction];
	}

	@Override
	public final QFunction getQFunction() {
		return qFunction;
	}

	public final void setEps(double eps) {
		if(eps < 0. || eps > 1.) {
			throw new IllegalArgumentException("eps must be within [0,1]");
		}
		this.eps = eps;
	}

	public final double[][] getProbaTable(double[][] xs) {
		final double[][] probas = new double[xs.length][nActions];
		for(int x=0; x<xs.length; x++) {
			computePolicyDistribution(xs[x]);
			ArrUtils.zeros(probas[x]);
			probas[x][indBestAction] = 1.;
		}
		return probas;
	}

	@Override
	public final String toString() {
		return "Epsilon-greedy policy over Q-function";
	}
}
