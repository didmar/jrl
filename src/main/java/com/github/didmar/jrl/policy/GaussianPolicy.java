package com.github.didmar.jrl.policy;

import org.eclipse.jdt.annotation.NonNull;

import com.github.didmar.jrl.utils.RandUtils;
import com.github.didmar.jrl.utils.array.ArrUtils;

/**
 * A policy based on a normal law. The standard variation of the noise,
 * which we call the exploratory noise, is fixed. The mean and its partial 
 * derivative over the policy parameters must be overridden.
 * 
 * @author Didier Marin
 */
public abstract class GaussianPolicy extends LogDifferentiablePolicy {

	/** State-space dimension */
	protected final int xDim;
	/** Action-space dimension */
	protected final int uDim;
	/** Action-space lower bound */
	protected final double[] uMin;
	/** Action-space upper bound */
	protected final double[] uMax;
	/** Std. dev. of the policy normal distribution */
	private final double[] sigma;
	/** Length of the policy parameters vector */
	protected final int n;
	/** State for which the policy distribution was computed */
	private final double[] distribX;
	/** Indicates if the mean of the normal distribution might have changed
	 * since we computed it : this flag will be raised if we change the policy
	 * parameters. */
	private boolean distribHasChanged;
	
	// arrays for temporary storage to avoid mem. alloc.
	/** Mean action according to the current distribution */
	private final double[] mu;
	/** Used to store the action */
	private final double[] uTmp;
	/** Used to store the log-derivative of the policy parameters */
	private final double[] der;
	/** Used to store the derivative of the mean of the normal distribution */
	private final double[] dermu;
	
	/**
	 * Constructor
	 * @param xDim State-space dimension
	 * @param sigma Std. dev. of the normal distribution
	 * @param uMin Action-space lower bound
	 * @param uMax Action-space upper bound
	 * @param thetaSize	Policy parameters length
	 * @throws IllegalArgumentException
	 */
	protected GaussianPolicy(int xDim, final double[] sigma,
			 			  final double[] uMin, final double[] uMax,
			 			  int thetaSize) {
		super(thetaSize);
		if(sigma.length <= 0) {
			throw new IllegalArgumentException("sigma must have a length greater than 0");
		}
		if(uMin.length != sigma.length) {
			throw new IllegalArgumentException("uMin must have the same length as sigma");
		}
		if(uMax.length != sigma.length) {
			throw new IllegalArgumentException("uMax must have the same length as sigma");
		}
		this.xDim = xDim;
		uDim = sigma.length;
		this.uMin = uMin;
		this.uMax = uMax;
		this.sigma = new double[sigma.length];
		setSigma(sigma);
		n = thetaSize;
		distribX = new double[xDim];
		// raise the flag since the current distribX is undefined
		distribHasChanged = true;
		
		// arrays for temporary storage to avoid mem. alloc.
		mu = new double[uDim];
		uTmp = new double[uDim];
		dermu = new double[n];
		der = new double[n];
	}
	
	/* (non-Javadoc)
	 * @see jrl.policy.Policy#drawAction()
	 */
	@NonNull
	public final double[] drawAction() {
		RandUtils.normal(mu,sigma,uTmp);
        ArrUtils.boundVector(uTmp,uMin,uMax);
        return uTmp;
	}
	
	/* (non-Javadoc)
	 * @see jrl.policy.Policy#computePolicyDistribution(double[])
	 */
	public final void computePolicyDistribution(@NonNull final double[] x) {
		assert x.length == xDim;
		
		// Compute the mean action and store it
		meanAction(x,mu);
		// Store of the associated state
		System.arraycopy(x, 0, distribX, 0, xDim);
		distribHasChanged = false;
	}
	
	/**
	 * Computes the mean action as a function of the state and put the result
	 * in a given array.
	 * @param x    the state to compute the mean action for
	 * @param result   the mean action
	 */
	public abstract void meanAction(final double[] x, final double[] result);
	
	/**
	 * Computes the partial derivative of the mean action function
	 * ({@link #meanAction(double[], double[])}) over the policy parameters and
	 * put the result in a given array.
	 * @param x        the state to compute the mean action derivative for
	 * @param result    the mean action
	 */
	public abstract void dMeanActiondTheta(final double[] x, final double[] result);
	
	/* (non-Javadoc)
	 * @see jrl.policy.ILogDifferentiablePolicy#dLogdTheta(double[], double[], int)
	 */
	@NonNull
	public final double[] dLogdTheta(@NonNull final double[] x, @NonNull final double[] u) {
		assert x.length == xDim;
		assert u.length == uDim;
		
		// Compute the distribution for this state, if not already done
        if((!ArrUtils.arrayEquals(x,distribX)) || distribHasChanged) {        	
            computePolicyDistribution(x);
        }
        // For each action dimension
        for(int i=0; i<uDim; i++) {
        	// Compute the action-dependent part of the log-derivative
        	double uMinMuDivSigma = (u[i] - mu[i]) / Math.pow(sigma[i],2);
        	// Compute the mean action derivative
        	dMeanActiondTheta(x, dermu);
        	// For each state feature, compute the log-derivative
        	for(int j=0; j<dermu.length; j++) {
        		der[i*dermu.length+j] = dermu[j] * uMinMuDivSigma;
        	}
        }
        return der;
	}
	
	/**
	 * Returns the std dev of the policy normal distribution.
	 * @return the std dev of the policy normal distribution
	 */
	public final double[] getSigma() {
		return sigma;
	}

	/**
	 * Set the std dev of the policy normal distribution.
	 * @param sigma    a std dev over the action space
	 */
	public final void setSigma(final double[] sigma) {
		System.arraycopy(sigma, 0, this.sigma, 0, sigma.length);
	}
	
	/* (non-Javadoc)
	 * @see jrl.policy.LogDifferentiablePolicy#setParams(double[])
	 */
	@Override
	public final void setParams(@NonNull final double[] theta) {
		distribHasChanged = true;
		super.setParams(theta);
	}

	/* (non-Javadoc)
	 * @see jrl.policy.LogDifferentiablePolicy#updateParams(double[])
	 */
	@Override
	public final void updateParams(@NonNull final double[] delta) {
		distribHasChanged = true;
		super.updateParams(delta);
	}

	@Override
	public String toString() {
		return "GaussianPolicy";
	}
}