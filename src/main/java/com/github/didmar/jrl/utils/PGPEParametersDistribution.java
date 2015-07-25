package com.github.didmar.jrl.utils;

import com.github.didmar.jrl.stepsize.StepSize;
import com.github.didmar.jrl.utils.array.ArrUtils;

/**
 * Gaussian parameters distribution which can be updated by an estimation
 * of its performance gradient.
 * @see com.github.didmar.jrl.agent.PGPE
 * @author Didier Marin
 */
public final class PGPEParametersDistribution
		extends GaussianParametersDistribution {
	
	private final double minSigma;
	/** Learning step */
	private final StepSize stepSize;
	
	// used for temporary storage
	private final double[] dJdMean;
	private final double[] dJdSigma;
	
	public PGPEParametersDistribution(double[] mean, double[] sigma,
			double minSigma, StepSize stepSize) {
		super(mean, sigma);
		if(minSigma <= 0) {
			throw new RuntimeException("minSigma must be greater than zero");
		}
		this.minSigma = minSigma;
		this.stepSize = stepSize;
		
		dJdMean = ArrUtils.zeros(n);
		dJdSigma = ArrUtils.zeros(n);
	}
	
	/**
	 * Update the distribution according to the gradient of its estimated
	 * performance.
	 * @param R      performance of each sample parameters
	 * @param params sample parameters
	 */
	public final void updateParamsDistribution(double[] R, double[][] params) {
		
		stepSize.updateStep();
		double alpha = stepSize.getStep();
		
		// Estimate the performance gradient of the distribution.
		ArrUtils.zeros(dJdMean);
		ArrUtils.zeros(dJdSigma);
		// For each sample parameters / episode
		for(int i=0; i<params.length; i++) {
			final double r = R[i]; 
			for(int j=0; j<n; j++) {
				final double paramsMinusMean = params[i][j]-mean[j];
				final double sigmaSqu = Math.pow(sigma[j],2);
				dJdMean[j] +=
					r * (paramsMinusMean / sigmaSqu);
				dJdSigma[j] +=
					r * ((Math.pow(paramsMinusMean,2) - sigmaSqu)
						 / (sigma[j]*sigmaSqu));
			}
		}
		
		//System.out.println("mean="+Utils.toString(mean));
		//System.out.println("sigma="+Utils.toString(sigma));
		
		// Update the distribution with this gradient
		for(int j=0; j<n; j++) {
			mean[j]  = alpha * dJdMean[j] / params.length;
			sigma[j] = Math.max(alpha * dJdSigma[j] / params.length, minSigma);
		}
		
		//System.out.println("mean="+Utils.toString(mean));
		//System.out.println("sigma="+Utils.toString(sigma));
	}
}
