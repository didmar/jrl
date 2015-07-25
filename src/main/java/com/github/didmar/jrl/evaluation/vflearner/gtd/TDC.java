package com.github.didmar.jrl.evaluation.vflearner.gtd;

import com.github.didmar.jrl.evaluation.valuefunction.LinearVFunction;
import com.github.didmar.jrl.evaluation.vflearner.td.TD;
import com.github.didmar.jrl.stepsize.StepSize;
import com.github.didmar.jrl.utils.DiscountFactor;
import com.github.didmar.jrl.utils.array.ArrUtils;

/**
 * Linear TD with gradient Correction, based on Sutton et al.
 * "Fast Gradient-Descent Methods for Temporal-Difference Learning with Linear
 * Function Approximation". Suitable for off-policy learning, i.e. with an
 * arbitrary stationary state distribution.
 *
 * @author Didier Marin
 */
public final class TDC extends TD {

	/** Secondary weight vector */
	private final double[] w;
	private final double betaOverAlphaRatio;
	private final StepSize alphaStep;
	private final int n;

	private final double[] delta;

	public TDC(LinearVFunction vFunction, DiscountFactor gamma,
			StepSize alphaStep, double betaOverAlphaRatio) {
		super(vFunction, gamma);
		this.alphaStep = alphaStep;
		this.betaOverAlphaRatio = betaOverAlphaRatio;
		n = vFunction.getParamsSize();
		w = new double[n];

		delta = new double[n];
	}

	@Override
	public final void addSample(double[] x, double[] xn, double r, boolean isTerminal) {
		// Update the step-size
		alphaStep.updateStep();
		final double alpha = alphaStep.getStep();
		final double beta  = betaOverAlphaRatio * alpha;
		// Compute the TD Error
		final double[] phix = ((LinearVFunction)vFunction).getFeatures().phi(x);
		// TODO if the sample is terminal, use phixn = zeros
		final double[] phixn = ((LinearVFunction)vFunction).getFeatures().phi(xn);
		final double tdErr = vFunction.tdError(x, xn, r, isTerminal, gamma);
		// Update approximation parameters
		double phixw = ArrUtils.dotProduct(phix,w,n);
		for(int i=0; i<n; i++) {
			delta[i] = alpha*(tdErr*phix[i] - gamma.value*phixn[i]*phixw);
		}
	    // Update the state value function
		((LinearVFunction)vFunction).updateParams(delta);
	    // Update w
		for(int i=0; i<n; i++) {
			w[i] += beta * (tdErr-phixw) * phix[i];
		}
	}

	@Override
	public final String toString() {
		return "TDC";
	}
}
