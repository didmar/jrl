package com.github.didmar.jrl.agent.ac;

import com.github.didmar.jrl.agent.LearningAgent;
import com.github.didmar.jrl.evaluation.valuefunction.LinearQFunction;
import com.github.didmar.jrl.evaluation.valuefunction.ParametricVFunction;
import com.github.didmar.jrl.evaluation.vflearner.ktd.KTDAV;
import com.github.didmar.jrl.features.CompatibleFeatures;
import com.github.didmar.jrl.policy.ILogDifferentiablePolicy;
import com.github.didmar.jrl.stepsize.StepSize;
import com.github.didmar.jrl.utils.DiscountFactor;

/**
 * Kalman Natural Actor-Critic is an Actor-Critic method that improves a
 * log-differentiable parameterized policy using its natural gradient.
 * The Critic computes a compatible approximation of the advantage and of the 
 * state value function, using {@link jrl_testing.evaluation.vflearner.ktd.KTDAV}.
 * 
 * @see jrl_testing.evaluation.vflearner.ktd.KTDAV
 * @author Didier Marin
 */
public final class KNAC extends LearningAgent {

	private final ILogDifferentiablePolicy pol;
	/** Learning step */
	private final StepSize stepSize;
	
	private final KTDAV ktdav;
	/** Linear advantage function approximator for the Critic part */
	private final LinearQFunction aFunction;
	
	// arrays for temporary storage to avoid mem. alloc.
	/** Used to store the policy update */
	private final double[] dJ;

	public KNAC(ILogDifferentiablePolicy pol, ParametricVFunction vFunction,
			StepSize stepSize, DiscountFactor gamma, DiscountFactor lambda,
			double P_evo_init, double eta, double P_obs_step, double k,
			double sigma_squared, int xDim, int uDim) throws Exception {
		super(pol);
		this.pol = pol;
		this.stepSize = stepSize;
		aFunction = new LinearQFunction(
				new CompatibleFeatures(pol, xDim, uDim), xDim, uDim);
		ktdav = new KTDAV(aFunction, vFunction, gamma, lambda, P_evo_init, eta, 
				P_obs_step,	k, sigma_squared);
		
		// arrays for temporary storage to avoid mem. alloc.
		dJ = new double[aFunction.getParamsSize()];
	}

	/* (non-Javadoc)
	 * @see jrl.environment.EnvironmentListener#newEpisode(double[], int)
	 */
	public final void newEpisode(double[] x0, int maxT) {
		// Nothing to do
	}

	/* (non-Javadoc)
	 * @see jrl.environment.EnvironmentListener#receiveSample(double[], double[], double[], double)
	 */
	public final void receiveSample(double[] x, double[] u, double[] xn, double r, boolean isTerminal) {
		// Critic part : transmit the sample to KTDAV
		ktdav.receiveSample(x, u, xn, r, isTerminal);
		// Actor part
		stepSize.updateStep();
		final double[] w = aFunction.getParams();
		final double beta = stepSize.getStep();
		for(int i=0; i<w.length; i++) {
			dJ[i] = beta * w[i];
		}
		pol.updateParams(dJ);
	}

	/* (non-Javadoc)
	 * @see jrl.environment.EnvironmentListener#endEpisode()
	 */
	public final void endEpisode() {
		// Nothing to do
	}
}
