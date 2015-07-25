package com.github.didmar.jrl.agent.ac;

import com.github.didmar.jrl.agent.LearningAgent;
import com.github.didmar.jrl.evaluation.valuefunction.LinearQFunction;
import com.github.didmar.jrl.evaluation.valuefunction.QFunction;
import com.github.didmar.jrl.evaluation.vflearner.td.AdvantageTDBootstrap;
import com.github.didmar.jrl.evaluation.vflearner.td.TD;
import com.github.didmar.jrl.features.CompatibleFeatures;
import com.github.didmar.jrl.policy.ILogDifferentiablePolicy;
import com.github.didmar.jrl.stepsize.StepSize;

// TODO debugger, le prob a l'air de venir du AdvantageTDBootstrap
/**
 * Vanilla Actor-Critic is variant of {@link BasicAC} that updates the policy
 * using an approximation of the advantage function instead of the TD error,
 * which reduces the variance of the performance gradient estimate. This
 * advantage function approximation is bootstraped from the TD error.
 * 
 * @see jrl_testing.evaluation.vflearner.td.AdvantageTDBootstrap
 * @author Didier Marin
 */
public final class VAC extends LearningAgent {
	
	/** The Critic part */	
	private final AdvantageTDBootstrap advTDBoot;
	/** Advantage function approximator for the Critic part */
	private final QFunction aFunction;
	/** Learning step */
	private final StepSize stepSize;
	/** Length of the policy parameters */
	private final int n;
	
	// arrays for temporary storage to avoid mem. alloc.
	/** Used the store the policy update */
	private final double[] dJ;
	
	/**
	 * Construct a {@link VAC}.
	 * @param pol    a policy which log is differentiable
	 * @param td     a TD leaner
	 * @param stepSize  a step-size for the policy parameters update
	 * @param xDim   the state-space dimension
	 * @param uDim   the action-space dimension
	 */
	public VAC(ILogDifferentiablePolicy pol, TD td, StepSize stepSize,
			int xDim, int uDim) {
		super(pol);
		this.stepSize = stepSize;
		aFunction = new LinearQFunction(new CompatibleFeatures(pol, xDim,
				uDim), xDim, uDim);
		advTDBoot = new AdvantageTDBootstrap(aFunction,td,stepSize); 
		n = pol.getParamsSize();
		
		// arrays for temporary storage to avoid mem. alloc.
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
        // Get the log policy gradient
        final double[] psi = ((ILogDifferentiablePolicy)pol).dLogdTheta(x,u);
        // Compute an estimation of the performance gradient (dJ ~ psi*A(x,u))
        // multiplied by the learning rate
        final double beta = stepSize.getStep();
        final double A = aFunction.get(x, u);
        for(int i=0; i<n; i++) {
        	dJ[i] = beta * psi[i] * A;
        }
        //System.out.println("dJ="+Utils.toString(dJ));
        // Update the policy parameters with it
        ((ILogDifferentiablePolicy)pol).updateParams(dJ);
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
		return "Vanilla Actor-Critic";
	}

}