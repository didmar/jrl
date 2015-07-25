package com.github.didmar.jrl.evaluation.vflearner.td;

import com.github.didmar.jrl.environment.EnvironmentListener;
import com.github.didmar.jrl.evaluation.valuefunction.QFunction;
import com.github.didmar.jrl.evaluation.vflearner.QFunctionLearner;
import com.github.didmar.jrl.stepsize.StepSize;
import com.github.didmar.jrl.utils.DiscountFactor;

// TODO Code a variant with eligibility traces (see Morimura's TDNAC)
/**
 * Learns the advantage function by bootstraping from a TD error.
 * @author Didier Marin
 */
public final class AdvantageTDBootstrap implements QFunctionLearner, EnvironmentListener {

	private final QFunction aFunction;
	private final TD tdLearner;
	/** Learning step */
	private final StepSize stepSize;
	private final DiscountFactor gamma;

	public AdvantageTDBootstrap(QFunction aFunction, TD tdLearner,
			StepSize stepSize) {
		// TODO check that the TDLeaner step-size decreases faster
		this.aFunction = aFunction;
		this.tdLearner = tdLearner;
		this.stepSize = stepSize;
		gamma = this.tdLearner.getDiscountFactor();
	}

	/* (non-Javadoc)
	 * @see jrl.environment.EnvironmentListener#newEpisode(double[], int)
	 */
	public final void newEpisode(double[] x0, int maxT) {
		// Transmit to the TD Learner
		tdLearner.newEpisode(x0, maxT);
	}

	/* (non-Javadoc)
	 * @see jrl.environment.EnvironmentListener#receiveSample(double[], double[], double[], double)
	 */
	public final void receiveSample(double[] x, double[] u, double[] xn, double r, boolean isTerminal) {
		// Transmit to the TD Learner
		tdLearner.receiveSample(x, u, xn, r, isTerminal);
		// Update the step-size
        stepSize.updateStep();
        // Compute the TD Error
        final double tdErr = tdLearner.getVFunction().tdError(x, xn, r, isTerminal, gamma);
        // Update the advantage function using the difference between the TD
        // error and the advantage value for the current state-action
        aFunction.updateForStateAction(x, u,
        		stepSize.getStep()*(tdErr-aFunction.get(x,u)));
	}
	
	/* (non-Javadoc)
	 * @see jrl.environment.EnvironmentListener#endEpisode()
	 */
	public final void endEpisode() {
		// Transmit to the TD Learner
		tdLearner.endEpisode();
	}

	/* (non-Javadoc)
	 * @see jrl.evaluation.vflearner.QFunctionLearner#getQFunction()
	 */
	public final QFunction getQFunction() {
		return aFunction;
	}

}
