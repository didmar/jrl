package com.github.didmar.jrl.agent;

import org.eclipse.jdt.annotation.NonNull;

import com.github.didmar.jrl.evaluation.valuefunction.LinearQFunction;
import com.github.didmar.jrl.evaluation.valuefunction.QFunction;
import com.github.didmar.jrl.evaluation.vflearner.SARSALambda;
import com.github.didmar.jrl.policy.QFunctionBasedPolicy;
import com.github.didmar.jrl.stepsize.StepSize;
import com.github.didmar.jrl.utils.DiscountFactor;

/**
 * A learning agent based on SARSA(lambda).
 *
 * @author Didier Marin
 */
public final class SARSAPolicyAgent extends LearningAgent {

	private final SARSALambda sarsa;

	public SARSAPolicyAgent(QFunctionBasedPolicy pol, DiscountFactor gamma,
			DiscountFactor lambda, StepSize stepSize) {
		super(pol);
		QFunction qFunction = pol.getQFunction();
		if(!(qFunction instanceof LinearQFunction)) {
			throw new IllegalArgumentException("The given policy must be " +
					"based on a LinearQFunction to use SARSA !");
		}
		sarsa = new SARSALambda((LinearQFunction)qFunction, pol, gamma, lambda, stepSize);
	}

	/* (non-Javadoc)
	 * @see jrl_testing.environment.EnvironmentListener#newEpisode(double[], int)
	 */
	public final void newEpisode(@NonNull final double[] x0, int maxT) {
		sarsa.newEpisode(x0, maxT);
	}

	/* (non-Javadoc)
	 * @see jrl_testing.environment.EnvironmentListener#receiveSample(double[], double[], double[], double, boolean)
	 */
	public final void receiveSample(@NonNull final double[] x,
									@NonNull final double[] u,
									@NonNull final double[] xn,
									double r, boolean isTerminal) {
		sarsa.receiveSample(x, u, xn, r, isTerminal);
	}

	/* (non-Javadoc)
	 * @see jrl_testing.environment.EnvironmentListener#endEpisode()
	 */
	public final void endEpisode() {
		sarsa.endEpisode();
	}

	@Override
	@NonNull
	public final String toString() {
		return "SARSA";
	}
}
