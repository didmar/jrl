package com.github.didmar.jrl.utils;

import org.eclipse.jdt.annotation.NonNull;

import com.github.didmar.jrl.utils.array.ArrUtils;

public class GaussianParametersDistribution {
	
	/** Mean of the parameters distribution */
	protected final double[] mean;
	/** Diagonal of the covariance matrix of the parameters distribution */
	protected final double[] sigma;
	/** Length of the parameter vector */
	protected final int n;
	
	public GaussianParametersDistribution(double[] mean, double[] sigma) {
		if(sigma.length != mean.length) {
			throw new IllegalArgumentException("mean and sigma must have the same length,"
					+" which is the number of parameters");
		}
		this.mean = ArrUtils.cloneVec(mean);
		this.sigma = ArrUtils.cloneVec(sigma);
		n = this.mean.length;
	}
		
	/**
	 * Draw nParams parameter vectors using the current Gaussian parameters
	 * distribution.
	 * Precondition : params must be a nParams-by-n matrix.
	 * @param params  matrix to put the sample parameter vectors in
	 */
	@SuppressWarnings("null")
	public final void drawParameters(double[][] params) {
		assert ArrUtils.hasShape(params, params.length, n);
		for(int i=0; i<params.length; i++) {
			RandUtils.normal(mean, sigma, params[i]);
		}
	}
	
	/**
	 * Draw nParams parameter vectors using the current Gaussian parameters
	 * distribution.
	 * @param nParams    number of parameter vectors to draw
	 * @return a nParams-by-n matrix containing the sample parameter vectors
	 */
	public final double[][] drawParameters(int nParams) {
		@NonNull final double[][] params = new double[nParams][n];
		drawParameters(params);
		return params;
	}
	
	/**
	 * Returns the mean of the parameters distribution
	 * @return mean of the parameters distribution
	 */
	public final double[] getMean() {
		return mean;
	}

	/**
	 * Returns the std. dev. of the parameters distribution
	 * @return std. dev. of the parameters distribution
	 */
	public final double[] getSigma() {
		return sigma;
	}
	
	/**
	 * Set the std. dev. of the parameters distribution
	 * @param std. dev. of the parameters distribution
	 */
	public final void setSigma(double[] sigma) {
		assert sigma.length == n;
		
		System.arraycopy(sigma, 0, this.sigma, 0, n);
	}
	
	/**
	 * Set the mean of the parameters distribution
	 * @param mean of the parameters distribution
	 */
	public final void setMean(double[] mean) {
		assert mean.length == n;
		
		System.arraycopy(mean, 0, this.mean, 0, n);
	}
}
