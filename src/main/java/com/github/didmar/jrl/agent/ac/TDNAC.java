package com.github.didmar.jrl.agent.ac;

import com.github.didmar.jrl.agent.LearningAgent;
import com.github.didmar.jrl.evaluation.valuefunction.LinearQFunction;
import com.github.didmar.jrl.evaluation.vflearner.td.AdvantageTDBootstrap;
import com.github.didmar.jrl.evaluation.vflearner.td.TD;
import com.github.didmar.jrl.features.CompatibleFeatures;
import com.github.didmar.jrl.policy.ILogDifferentiablePolicy;
import com.github.didmar.jrl.stepsize.StepSize;
import com.github.didmar.jrl.utils.DiscountFactor;

/**
 * TD-based Natural Actor-Critic is a variant of {@link VAC} that uses
 * the natural gradient.
 * It can be found in Morimura et al. 2005 "Utilizing the natural
 *  gradient in temporal difference reinforcement learning with eligibility
 *  traces" and algorithm 4 of Bhatnagar et al. 2007 "Natural-gradient 
 *  Actor-Critic algorithms".
 * 
 * @author Didier Marin
 */
public final class TDNAC extends LearningAgent {

	/** The Critic part */
	private final AdvantageTDBootstrap advTDBoot;
	/** Linear advantage function approximator for the Critic part */
	private final LinearQFunction aFunction;
	/** Learning step */
	private final StepSize stepSize;
	/** Forget factor */
	private final DiscountFactor kappa;
	/** Length of the policy parameters */
	private final int n;
	
	// arrays for temporary storage to avoid mem. alloc.
	private final double[] w;
	private final double[] dJ;
	
	/**
	 * Construct a {@link TDNAC}.
	 * @param pol    a policy which log is differentiable
	 * @param td     a TD leaner
	 * @param stepSize  a step-size for the policy parameters update
	 * @param kappa  forget factor
	 * @param xDim   the state-space dimension
	 * @param uDim   the action-space dimension
	 */
	public TDNAC(ILogDifferentiablePolicy pol, TD td, StepSize stepSize,
			DiscountFactor kappa, int xDim, int uDim) {
		super(pol);
		this.stepSize = stepSize;
		this.kappa = kappa;
		aFunction = new LinearQFunction(new CompatibleFeatures(pol, xDim,
				uDim), xDim, uDim);
		advTDBoot = new AdvantageTDBootstrap(aFunction,td,stepSize); 
		n = pol.getParamsSize();
		
		// arrays for temporary storage to avoid mem. alloc.
		w = new double[n];
		dJ = new double[n];
	}

	/* (non-Javadoc)
	 * @see jrl.environment.EnvironmentListener#newEpisode(double[], int)
	 */
	public final void newEpisode(double[] x0, int maxT) {
		// Transmit to the advantage function learner
		advTDBoot.newEpisode(x0,maxT);
	}

	/* (non-Javadoc)
	 * @see jrl.environment.EnvironmentListener#receiveSample(double[], double[], double[], double)
	 */
	public final void receiveSample(double[] x, double[] u, double[] xn, double r, boolean isTerminal) {
		// Transmit to the advantage function learner
		advTDBoot.receiveSample(x, u, xn, r, isTerminal);
		// Update the step-size
        stepSize.updateStep();
        // Compute an estimation of the performance gradient dJ as the
        // compatible advantage function approximation parameters w multiplied
        // by the learning rate
        final double beta = stepSize.getStep();
        System.arraycopy(aFunction.getParams(),0,w,0,n);
        for(int i=0; i<n; i++) {
        	dJ[i] = beta * w[i];
        }
        // Update the policy parameters with it
        ((ILogDifferentiablePolicy)pol).updateParams(dJ);
        // Partially forget w
        for(int i=0; i<n; i++) {
        	w[i] = kappa.value * w[i];
        }
        aFunction.setParams(w);
	}
	
	/* (non-Javadoc)
	 * @see jrl.environment.EnvironmentListener#endEpisode()
	 */
	public final void endEpisode() {
		// Transmit to the advantage function learner
		advTDBoot.endEpisode();
	}

	/* (non-Javadoc)
	 * @see java.lang.Object#toString()
	 */
	@Override
	public final String toString() {
		return "TD-based Natural Actor-Critic";
	}
}
