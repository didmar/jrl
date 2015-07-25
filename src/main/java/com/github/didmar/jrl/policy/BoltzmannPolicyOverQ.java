package com.github.didmar.jrl.policy;

import com.github.didmar.jrl.evaluation.valuefunction.QFunction;
import com.github.didmar.jrl.utils.RandUtils;
import com.github.didmar.jrl.utils.array.ArrUtils;

/**
 * Policy based on a Boltzmann distribution over a set of sample actions, using
 * a Q-value to weight the probability of each action.
 * @author Didier Marin
 */
public final class BoltzmannPolicyOverQ implements QFunctionBasedPolicy {

	/** State-action value function to compute the action distribution */
	private final QFunction qFunction;
	/** Set of possible actions */
	private final double[][] actions;
	/** Number of possible actions */
	private final int nActions;
	/** Temperature of the Boltzmann distribution. The higher the temperature,
	 * the more equiprobable the actions */
	private double temp;
	/** Action-space dimension */
	private final int uDim;
	/** Boltzmann distribution over sample actions */
	private final double[] prob;

	// arrays for temporary storage to avoid mem. alloc.
	/** Used to store the Q-value for each sample action */
	private final double[] QValues;

	public BoltzmannPolicyOverQ(QFunction qFunction, double[][] actions,
			double temp) {
		if(temp <= 0.) {
			throw new IllegalArgumentException("temp must be greater than 0");
		}
		this.qFunction = qFunction;
		this.actions = actions;
		nActions = this.actions.length;
		this.temp = temp;
		uDim = this.qFunction.getUDim();
		for(int i=0; i<nActions; i++) {
			if(uDim != this.actions[i].length) {
				throw new IllegalArgumentException("Some sample action " +
						"dimension is not compatible with the Q-function " +
						"action dimension");
			}
		}
		prob = new double[nActions];

		// arrays for temporary storage to avoid mem. alloc.
		QValues = new double[nActions];
	}

	/* (non-Javadoc)
	 * @see jrl.policy.Policy#computePolicyDistribution(double[])
	 */
	@Override
	public final void computePolicyDistribution(double[] x) {
		// Get the Q-value for each sample action
        for(int i=0; i<nActions; i++){
        	QValues[i] = qFunction.get(x, actions[i]);
        }
        // Add the min to all Q-values
        double QMin = ArrUtils.min(QValues);
        for(int i=0; i<nActions; i++){
        	QValues[i] += QMin;
        }
        // Normalize Q-values
        ArrUtils.norm(QValues);
        // For each sample action
        for(int i=0; i<nActions; i++){
	        // Compute the sample action weight
        	prob[i] = Math.exp(QValues[i]/temp);
        	if(Double.isNaN(prob[i])) {
        		prob[i] = Double.MAX_VALUE / (nActions+1);
        	} else {
        		prob[i] = Math.min(prob[i], Double.MAX_VALUE / (nActions+1));
        	}
        }
    	// Normalize the weights to get a Boltzmann
        // distribution over sample actions
        if(!ArrUtils.normalize(prob)) {
        	for(int i=0; i<nActions; i++) {
        		prob[i] = 1./((double)nActions);
        	}
        }
	}

	/* (non-Javadoc)
	 * @see jrl.policy.Policy#drawAction()
	 */
	@Override
	public final double[] drawAction() {
		// Draw an action according to the current Boltzmann distribution
		final int ind = RandUtils.drawFromDiscreteProbTable(prob);
		return actions[ind];
	}

	@Override
	public final QFunction getQFunction() {
		return qFunction;
	}

	public final void setTemp(double temp) {
		this.temp = temp;
	}

	public final double[][] getProbaTable(double[][] xs) {
		final double[][] probas = new double[xs.length][nActions];
		for(int x=0; x<xs.length; x++) {
			computePolicyDistribution(xs[x]);
			System.arraycopy(prob, 0, probas[x], 0, nActions);
		}
		return probas;
	}

	@Override
	public final String toString() {
		return "Boltzmann policy over Q-function";
	}
}
