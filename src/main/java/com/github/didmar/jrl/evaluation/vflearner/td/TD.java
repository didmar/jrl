package com.github.didmar.jrl.evaluation.vflearner.td;

import com.github.didmar.jrl.environment.EnvironmentListener;
import com.github.didmar.jrl.evaluation.valuefunction.VFunction;
import com.github.didmar.jrl.evaluation.vflearner.VFunctionLearner;
import com.github.didmar.jrl.utils.DiscountFactor;

/**
 * Prototype of state value function learners based on the TD error, in the
 * discounted reward case.
 * @author Didier Marin
 */
public abstract class TD implements VFunctionLearner, EnvironmentListener {

	/** State value function approximation */
	protected final VFunction vFunction;
	/** Discount factor */
	protected final DiscountFactor gamma;

	/**
	 * @param vFunction
	 * @param gamma
	 */
	public TD(VFunction vFunction, DiscountFactor gamma) {
		if(vFunction == null) {
			throw new IllegalArgumentException("vFunction must be non-null");
		}
		this.vFunction = vFunction;
		this.gamma = gamma;
	}

	public abstract void addSample(double[] x, double[] xn, double r, boolean isTerminal);
	
	/* (non-Javadoc)
	 * @see jrl.environment.EnvironmentListener#newEpisode(double[], int)
	 */
	public void newEpisode(double[] x0, int maxT) {
		// Nothing to do
	}

	/* (non-Javadoc)
	 * @see jrl.environment.EnvironmentListener#receiveSample(double[], double[], double[], double)
	 */
	public final void receiveSample(double[] x, double[] u, double[] xn, double r, boolean isTerminal) {
		addSample(x, xn, r, isTerminal);
	}

	/* (non-Javadoc)
	 * @see jrl.environment.EnvironmentListener#endEpisode()
	 */
	public final void endEpisode() {
		// Nothing to do
	}

	/* (non-Javadoc)
	 * @see jrl.evaluation.vflearner.VFunctionLearner#getVFunction()
	 */
	public final VFunction getVFunction() {
		return vFunction;
	}
	
	public final DiscountFactor getDiscountFactor() {
		return gamma;
	}
}
