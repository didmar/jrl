package com.github.didmar.jrl.agent.ac;

import com.github.didmar.jrl.agent.LearningAgent;
import com.github.didmar.jrl.evaluation.vflearner.td.TD;
import com.github.didmar.jrl.policy.ILogDifferentiablePolicy;
import com.github.didmar.jrl.stepsize.StepSize;
import com.github.didmar.jrl.utils.DiscountFactor;

/**
 * Very basic Actor-Critic architecture which forwards the samples to a TD
 * learner (Critic part) and uses the TD error to update its policy (Actor part).
 * The algorithm can be found as Algorithm 1 in Bhatnagar et al. 2007
 * "Natural-gradient actor-critic algorithms", where the average reward
 * is used instead of the discounted reward.
 *
 * @see jrl_testing.evaluation.vflearner.td.TD
 * @author Didier Marin
 */
public final class BasicAC extends LearningAgent {

	private final TD tdLearner;
	/** Learning step */
	private final StepSize stepSize;
	/** Discount factor */
	private final DiscountFactor gamma;
	/** Length of policy parameters */
	private final int n;

	// arrays for temporary storage to avoid mem. alloc.
	/** Used the store the policy update */
	private final double[] dJ;

	public BasicAC(ILogDifferentiablePolicy pol, TD tdLearner, StepSize stepSize) {
		super(pol);
		this.tdLearner = tdLearner;
		this.stepSize = stepSize;
		gamma = this.tdLearner.getDiscountFactor();
		n = pol.getParamsSize();
		// TODO check that the TD learner has a faster step size !

		dJ = new double[n];
	}

	/* (non-Javadoc)
	 * @see jrl.environment.EnvironmentListener#newEpisode(double[], int)
	 */
	public final void newEpisode(double[] x0, int maxT) {
		// Transmit to the TD Learner
		tdLearner.newEpisode(x0,maxT);
	}

	/* (non-Javadoc)
	 * @see jrl.environment.EnvironmentListener#receiveSample(double[], double[], double[], double)
	 */
	public final void receiveSample(double[] x, double[] u, double[] xn, double r, boolean isTerminal) {
		// Transmit to the TD Learner
		tdLearner.receiveSample(x, u, xn, r, isTerminal);
		// Update the step-size
        stepSize.updateStep();
        // Get the TD Error
        final double tdErr = tdLearner.getVFunction().tdError(x,xn,r,isTerminal,gamma);
        // Get the log policy gradient
        final double[] psi = ((ILogDifferentiablePolicy)pol).dLogdTheta(x, u);
        // Compute an estimation of the performance gradient (dJ ~ psi * tdErr)
        // multiplied by the learning rate
        final double beta = stepSize.getStep();
        for(int i=0; i<n; i++) {
        	dJ[i] = beta * psi[i] * tdErr;
        }
        // Update the policy parameters with it
        ((ILogDifferentiablePolicy)pol).updateParams(dJ);
	}

	/* (non-Javadoc)
	 * @see jrl.environment.EnvironmentListener#endEpisode()
	 */
	public final void endEpisode() {
		// Transmit to the TD Learner
		tdLearner.endEpisode();
	}

	/* (non-Javadoc)
	 * @see java.lang.Object#toString()
	 */
	@Override
	public final String toString() {
		return "Basic Actor-Critic";
	}

}
