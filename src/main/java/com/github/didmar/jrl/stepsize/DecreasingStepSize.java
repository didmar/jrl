package com.github.didmar.jrl.stepsize;

/**
 * A simple decreasing step-size. Suitable for stochastic gradient descent and
 * TD algorithms.
 * 
 * @author Didier Marin
 */
public class DecreasingStepSize implements StepSize {

	private double alpha;
	private final double alpha0;
	private final double alphac;
	private final double tpow;
	private int currentT;
	
	public DecreasingStepSize(double alpha0, double alphac, double tpow) {
		this.alpha0 = alpha0;
		this.alphac = alphac;
		this.tpow = tpow;
		currentT = 0;
	}
	
	public DecreasingStepSize(double alpha0, double alphac) {
		this.alpha0 = alpha0;
		this.alphac = alphac;
		this.tpow = 1.;
		currentT = 0;
	}
	
	public DecreasingStepSize() {
		alpha0 = 0.01;
		alphac = 10000;
		tpow = 1.;
		currentT = 0;
	}

	/* (non-Javadoc)
	 * @see jrl.stepsize.StepSize#updateStep()
	 */
	public final void updateStep() {
		stepAtTime(currentT);
		currentT++;
	}

	/* (non-Javadoc)
	 * @see jrl.stepsize.StepSize#stepAtTime(int)
	 */
	public final void stepAtTime(int t) {
		alpha = alpha0 * alphac / (alphac + Math.pow(t,tpow));
	}

	/* (non-Javadoc)
	 * @see jrl.stepsize.StepSize#getStep()
	 */
	public final double getStep() {
		return alpha;
	}
	
	public final double getAlpha0() {
		return alpha0;
	}
	
	public final double getAlphaC() {
		return alphac;
	}

	@Override
	public final String toString() {
		return "DecreasingStepSize [alpha=" + alpha + ", alpha0=" + alpha0
				+ ", alphac=" + alphac + ", tpow=" + tpow + "]";
	}
	
	/**
	 * Returns an array that contains two {@link DecreasingStepSize}, the first
	 * being faster than the second. These are suitable for incremental
	 * Actor-Critic methods : the Critic uses the first one and the Actor the
	 * second.
	 * @param alpha0
	 * @param alphac
	 * @param beta0
	 * @param betac
	 * @return an array that contains two {@link DecreasingStepSize}, the first
	 *         being faster than the second
	 */
	public static final DecreasingStepSize[] createTwoTimescaleStepSizes(
			double alpha0, double alphac, double beta0, double betac) {
		final DecreasingStepSize[] steps = {
				new DecreasingStepSize(alpha0, alphac, 1.),
				new DecreasingStepSize(beta0, betac, 2./3.)
			};
		return steps;
	}
}
