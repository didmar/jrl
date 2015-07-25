package com.github.didmar.jrl.evaluation.vflearner;

import com.github.didmar.jrl.environment.EnvironmentListener;
import com.github.didmar.jrl.evaluation.valuefunction.LinearQFunction;
import com.github.didmar.jrl.evaluation.valuefunction.QFunction;
import com.github.didmar.jrl.policy.Policy;
import com.github.didmar.jrl.stepsize.StepSize;
import com.github.didmar.jrl.utils.array.ArrUtils;

// TODO Bug corrigé, à retester
// TODO Implement eligibility traces
/**
 * GQ(lambda) as in Maei et al. 2010 "GQ(lambda): A general gradient algorithm
 * for temporal-difference prediction learning with eligibility traces".
 * @author Didier Marin
 */
public final class GQLambda implements QFunctionLearner, EnvironmentListener {

	/** State action value function approximator */
	private final LinearQFunction qFunction;
	/** Policy being evaluated */
	private final Policy pol;
	/** Secondary weight vector */
	private final double[] w;
	/** Reward discount factor */
	private final double gamma;
	/** Eligibility factor */
	@SuppressWarnings("unused")
	private final double lambda;
	private final StepSize alphaStep;
	private final double betaOverAlphaRatio;
	/** State-space dimension */
	private final int xDim;
	/** Action-space dimension */
	private final int uDim;
	private final int n;
	
	private final double[] xu;
	private final double[] xnun;
	private final double[] delta;
	
	public GQLambda(LinearQFunction qFunction, Policy pol, double gamma,
			double lambda, StepSize alphaStep, double betaOverAlphaRatio) {
		if(gamma < 0. || gamma > 1.) {
			throw new IllegalArgumentException("gamma must be in [0,1]");
		}
		if(lambda < 0. || lambda > 1.) {
			throw new IllegalArgumentException("lambda must be in [0,1]");
		}
		this.qFunction = qFunction;
		this.pol = pol;
		this.gamma = gamma;
		this.lambda = lambda;
		this.alphaStep = alphaStep;
		this.betaOverAlphaRatio = betaOverAlphaRatio;
		n = qFunction.getParamsSize();
		w = ArrUtils.zeros(n);
		xDim = qFunction.getXDim();
		uDim = qFunction.getUDim();
		
		xu = new double[xDim+uDim];
		xnun = new double[xDim+uDim];
		delta = new double[n];
	}
	
	/* (non-Javadoc)
	 * @see jrl.evaluation.vflearner.QFunctionLearner#getQFunction()
	 */
	@Override
	public final QFunction getQFunction() {
		return qFunction;
	}

	@Override
	public final void endEpisode() {
		// Nothing to do
	}

	@Override
	public final void newEpisode(double[] x0, int maxT) {
		// Nothing to do
	}

	@Override
	public final void receiveSample(double[] x, double[] u, double[] xn, double r, boolean isTerminal) {
		// Update the step-size
		alphaStep.updateStep();
		final double alpha = alphaStep.getStep();
		final double beta  = betaOverAlphaRatio * alpha;
		
		System.arraycopy(x, 0, xu, 0, xDim);
		System.arraycopy(u, 0, xu, xDim, uDim);
		
		pol.computePolicyDistribution(xn);
		System.arraycopy(xn, 0, xnun, 0, xDim);
		System.arraycopy(pol.drawAction(), 0, xnun, xDim, uDim);
		
		// Compute the TD Error
		final double[] psix = qFunction.getFeatures().phi(xu);
		final double[] theta = qFunction.getParams();
		double tdErr = r - ArrUtils.dotProduct(psix, theta, n);
		double[] psixn = null;
		if(!isTerminal) {
			psixn = qFunction.getFeatures().phi(xnun);
			tdErr += gamma * ArrUtils.dotProduct(psixn, theta, n);
		}
		// Update approximation parameters
		double psixw = ArrUtils.dotProduct(psix,w,n);
		if(!isTerminal) {
			for(int i=0; i<n; i++) {
				delta[i] = alpha*(tdErr*psix[i] - gamma*psixn[i]*psixw);
			}
		} else {
			for(int i=0; i<n; i++) {
				delta[i] = alpha*tdErr*psix[i];
			}
		}
	    // Update the state value function
		qFunction.updateParams(delta);
	    // Update w
		for(int i=0; i<n; i++) {
			w[i] += beta * (tdErr-psixw) * psix[i];
		}
		
	}

}
