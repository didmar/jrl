package com.github.didmar.jrl.stepsize;

/**
 * A constant step-size.
 * @author Didier Marin
 *
 */
public final class ConstantStepSize implements StepSize {

	private final double alpha;
	
	public ConstantStepSize(double alpha) {
		this.alpha = alpha;
		// assert alpha > 0.
	}
	
	/* (non-Javadoc)
	 * @see jrl.environment.stepsize.StepSize#updateStep()
	 */
	public final void updateStep() {
		// Nothing to compute
	}

	/* (non-Javadoc)
	 * @see jrl.environment.stepsize.StepSize#stepAtTime(int)
	 */
	public final void stepAtTime(int t) {
		// Nothing to do
	}

	public final double getAlpha() {
		return alpha;
	}

	/* (non-Javadoc)
	 * @see jrl.stepsize.StepSize#getStep()
	 */
	public final double getStep() {
		return alpha;
	}

	@Override
	public final String toString() {
		return "ConstantStepSize [alpha=" + alpha + "]";
	}	
}
