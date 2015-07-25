package com.github.didmar.jrl.agent;

import org.eclipse.jdt.annotation.NonNull;

import com.github.didmar.jrl.evaluation.valuefunction.LinearQFunction;
import com.github.didmar.jrl.evaluation.valuefunction.QFunction;
import com.github.didmar.jrl.evaluation.vflearner.QLearning;
import com.github.didmar.jrl.policy.QFunctionBasedPolicy;
import com.github.didmar.jrl.stepsize.StepSize;
import com.github.didmar.jrl.utils.DiscountFactor;

/**
 * A learning agent based on Q(lambda).
 *
 * @author Didier Marin
 */
public final class QLearningPolicyAgent extends LearningAgent {

	private final QLearning qLearning;

	public QLearningPolicyAgent(QFunctionBasedPolicy pol, double[][] actions,
			DiscountFactor gamma, DiscountFactor lambda, StepSize stepSize) {
		super(pol);
		final QFunction qFunction = pol.getQFunction();
		if(!(qFunction instanceof LinearQFunction)) {
			throw new IllegalArgumentException("The given policy must be"
					+" based on a "+LinearQFunction.class.getName()
					+" to use Q-Learning !");
		}
		qLearning = new QLearning((LinearQFunction)qFunction, actions, gamma,
				lambda, stepSize);
	}

	/* (non-Javadoc)
	 * @see jrl_testing.environment.EnvironmentListener#newEpisode(double[], int)
	 */
	public final void newEpisode(double[] x0, int maxT) {
		qLearning.newEpisode(x0, maxT);
	}

	/* (non-Javadoc)
	 * @see jrl_testing.environment.EnvironmentListener#receiveSample(double[], double[], double[], double, boolean)
	 */
	public final void receiveSample(double[] x, double[] u, double[] xn, double r, boolean isTerminal) {
		qLearning.receiveSample(x, u, xn, r, isTerminal);
	}

	/* (non-Javadoc)
	 * @see jrl_testing.environment.EnvironmentListener#endEpisode()
	 */
	public final void endEpisode() {
		qLearning.endEpisode();
	}

	@Override
	@NonNull
	public final String toString() {
		return "Q-Learning";
	}
}
