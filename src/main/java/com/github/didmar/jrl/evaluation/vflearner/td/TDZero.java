package com.github.didmar.jrl.evaluation.vflearner.td;

import com.github.didmar.jrl.evaluation.valuefunction.LinearVFunction;
import com.github.didmar.jrl.stepsize.StepSize;
import com.github.didmar.jrl.utils.DiscountFactor;

/**
 * Implementation of TD(0).
 * @author Didier Marin
 */
public class TDZero extends TD {

	/** Learning step */
	protected StepSize stepSize;
	
	/**
	 * @param vFunction
	 * @param gamma
	 */
	public TDZero(LinearVFunction vFunction, StepSize stepSize,
			DiscountFactor gamma) {
		super(vFunction, gamma);
		this.stepSize = stepSize;
	}

	/* (non-Javadoc)
	 * @see jrl.evaluation.valuefunction.VFunctionLearner#addSample(double[], double[], double, boolean)
	 */
	@Override
	public void addSample(double[] x, double[] xn, double r, boolean isTerminal) {
		// Update the step-size
		stepSize.updateStep();
		// Compute the TD Error
		final double tdErr = vFunction.tdError(x, xn, r, isTerminal, gamma);
		// Update the state value function with the TD Error
		((LinearVFunction)vFunction).updateForState(x, stepSize.getStep()*tdErr);
	}

	@Override
	public String toString() {
		return "TD(0)";
	}
}
