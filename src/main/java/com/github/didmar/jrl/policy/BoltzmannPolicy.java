package com.github.didmar.jrl.policy;

import org.eclipse.jdt.annotation.NonNull;

import com.github.didmar.jrl.features.Features;
import com.github.didmar.jrl.utils.RandUtils;
import com.github.didmar.jrl.utils.array.ArrUtils;

// FIXME when temp is too close to zeros, the actions become equiprobable !
// TODO replace by BoltzmannPolicyOverQ ?
/**
 * Boltzmann distribution over a set of sample actions, using a weighted sum of
 * state-action features. This policy is suitable for discrete environments,
 * using the set of all possible actions (i.e. {0, 1, ... uCard-1}).
 * 
 * @author Didier Marin
 */
public final class BoltzmannPolicy extends LogDifferentiablePolicy {

	/** State-action features */
	private final Features stateActionFeatures;
	/** Set of possible actions */
	private final double[][] actions;
	/** Number of possible actions */
	private final int nActions;
	/** Temperature of the Boltzmann distribution. The higher the temperature,
	 * the more equiprobable the actions */
	private double temp;
	/** Upper bound for the policy parameters, based on the temperature */
	private final double thetaMax;
	/** State-space dimension */
	private final int xDim;
	/** Action-space dimension */
	private final int uDim;
	/** Boltzmann distribution over sample actions */
	private final double[] prob;
	/** State for which the policy distribution was computed */
	private final double[] distribX;
	/** Indicates if the mean of the normal distribution might have changed
	 * since we computed it : this flag will be raised if we change the policy
	 * parameters. */
	private boolean distribHasChanged;
	
	// arrays for temporary storage to avoid mem. alloc.
	/** Used to concatenate a state and an action */
	private final double[] xu;
	/** Used to store the state-action features ouput for each sample action */ 
	private final double[][] phis;
	/** Used to store the log policy gradient */
	private final double[] der;
	/** Used to store the distance between a given action and each sample
	 * action */
	private final double[] distToSampleAction;
	
	/**
	 * Construct a {@link BoltzmannPolicy}.
	 * Preconditions :
	 *     stateActionFeatures must be normalized ({@link Features#isNormalized()}
	 *     returns true),
	 *     temp must be greater than 0
	 * @param stateActionFeatures  normalized state-action features
	 * @param actions              array of the possible actions
	 * @param temp                 temperature of the Boltzmann distribution.
	 * @throws IllegalArgumentException
	 */
	public BoltzmannPolicy(final Features stateActionFeatures,
						   final double[][] actions,
						   double temp) {
		super(stateActionFeatures.outDim);
		if(temp <= 0.) {
			throw new IllegalArgumentException("temp must be greater than 0");
		}
		if(!stateActionFeatures.isNormalized()) {
			throw new IllegalArgumentException("State-action features must be "
					+"normalized");
		}
		if(!ArrUtils.isMatrix(actions)) {
			throw new IllegalArgumentException("actions array must be a matrix");
		}
		this.stateActionFeatures = stateActionFeatures;
		this.actions = actions;
		nActions = this.actions.length;
		this.temp = temp;
		thetaMax = Math.log(1e20)*this.temp;
		uDim = this.actions[0].length;
		xDim = this.stateActionFeatures.inDim - uDim;
		if(xDim <= 0) {
			throw new IllegalArgumentException("State-action features input "
					+"dimension should be greater than sample actions dimension");
		}
		prob = new double[nActions];
		distribX = new double[xDim];
		// raise the flag since the current distribX is undefined
		distribHasChanged = true;
		
		// arrays for temporary storage to avoid mem. alloc.
		xu = new double[xDim+uDim];
		phis = new double[nActions][getParamsSize()];
		der = new double[getParamsSize()];
		distToSampleAction = new double[nActions]; 
	}

	/* (non-Javadoc)
	 * @see jrl.policy.ILogDifferentiablePolicy#dLogdTheta(double[], double[])
	 */
	@SuppressWarnings("null")
	@NonNull
	public final double[] dLogdTheta(@NonNull final double[] x,
									 @NonNull final double[] u) {
		// Compute the distribution for this state, if not already done
        if((!ArrUtils.arrayEquals(x,distribX)) || distribHasChanged) {        	
            computePolicyDistribution(x);
        }
        // Find u in the sample actions
        int ind = -1;
        for(int i=0; i<nActions; i++) {
        	if(ArrUtils.arrayEquals(u, actions[i])) {
        		ind = i;
        		break;
        	}
        }
        // If u is not a sample action, get the closest sample action in L2 norm
        if(ind == -1) {
        	for(int i=0; i<nActions; i++) {
        		distToSampleAction[i] = 0.;
        		for(int j=0; j<uDim; j++) {
        			distToSampleAction[i] += Math.pow(actions[i][j] - u[j], 2);
        		}
        	}
        	ind = ArrUtils.argmin(distToSampleAction);
        }
        // Compute the derivative
        for(int i=0; i<getParamsSize(); i++) {
        	der[i] = phis[ind][i];
        	for(int j=0; j<nActions; j++) {
        		der[j] -= phis[j][i] * prob[j];
        	}
        }
        return der;
	}

	/* (non-Javadoc)
	 * @see jrl.policy.Policy#computePolicyDistribution(double[])
	 */
	@SuppressWarnings("null")
	public final void computePolicyDistribution(@NonNull final double[] x) {
		// For each sample action
        for(int i=0; i<nActions; i++){
        	// Concatenate the state and the sample action
        	System.arraycopy(x, 0, xu, 0, xDim);
        	System.arraycopy(actions[i], 0, xu, xDim, uDim);
            // Get the state-action features for this action
            stateActionFeatures.phi(xu, phis[i]);
            // Compute the sample action weight
        	prob[i] = Math.exp(ArrUtils.dotProduct(phis[i], theta, phis[i].length)/temp);
        }
        // Normalize the weights to get a Boltzmann
        // distribution over sample actions
        ArrUtils.normalize(prob);
        // Store of the associated state
		System.arraycopy(x, 0, distribX, 0, xDim);
		distribHasChanged = false;
	}

	/* (non-Javadoc)
	 * @see jrl.policy.Policy#drawAction()
	 */
	@SuppressWarnings("null")
	@NonNull
	public final double[] drawAction() {
		// Draw an action according to the current Boltzmann distribution
		final int ind = RandUtils.drawFromDiscreteProbTable(prob);
        return actions[ind];
	}
	
	@SuppressWarnings("null")
	public final double[][] getProbaTable(final double[][] xs) {
		@NonNull final double[][] probas = new double[xs.length][nActions];
		for(int x=0; x<xs.length; x++) {
			computePolicyDistribution(xs[x]);
			System.arraycopy(prob, 0, probas[x], 0, nActions);
		}
		return probas;
	}
	
	/* (non-Javadoc)
	 * @see jrl.policy.LogDifferentiablePolicy#setParams(double[])
	 */
	@Override
	public final void setParams(final double[] theta) {
		distribHasChanged = true;
		super.setParams(theta);
	}

	/* (non-Javadoc)
	 * @see jrl.policy.LogDifferentiablePolicy#updateParams(double[])
	 */
	@Override
	public final void updateParams(final double[] delta) {
		distribHasChanged = true;
		super.updateParams(delta);
	}

	/* (non-Javadoc)
	 * @see jrl.policy.LogDifferentiablePolicy#boundParams(double[])
	 */
	@Override
	public final boolean boundParams(final double[] params) {
		return ArrUtils.boundVector(params, 0, thetaMax);
	}
	
	public final void setTemp(double temp) {
		this.temp = temp;
	}
}
