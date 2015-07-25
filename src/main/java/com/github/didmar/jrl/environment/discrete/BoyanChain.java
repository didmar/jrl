package com.github.didmar.jrl.environment.discrete;

import com.github.didmar.jrl.utils.RandUtils;

/**
 * Uncontrolled stochastic environment designed to test value function
 * approximation methods. This implementation has 14 discrete state.
 * See iLSTD paper for further details.
 *
 * @author Didier Marin
 */
public final class BoyanChain extends DiscreteEnvironment {

	public static final int CHAIN_LENGTH = 14;

	public BoyanChain() {
		super(CHAIN_LENGTH,1);
	}

	/* (non-Javadoc)
	 * @see jrl.environment.Environment#isTerminal(double[], double[], double[])
	 */
	@Override
	public final boolean isTerminal(double[] x, double[] u, double[] xn) {
		assert x.length == xDim;
		assert u.length == uDim;
		assert xn.length == xDim;

		return (x[0]==0 ? true : false);
	}

	/* (non-Javadoc)
	 * @see jrl.environment.Environment#nextState(double[], double[])
	 */
	@Override
	public final double[] nextState(double[] x, double[] u) {
		assert x.length == xDim;
		assert u.length == uDim;

		if(x[0] >= 3) {
			return new double[]{x[0]-1-(RandUtils.nextInt(2))};
		} else if (x[0] == 2) {
			return new double[]{1};
		}
		return new double[]{0}; // absorbing state
	}

	/* (non-Javadoc)
	 * @see jrl.environment.Environment#reward(double[], double[], double[])
	 */
	@Override
	public final double reward(double[] x, double[] u, double[] xn) {
		assert x.length == xDim;
		assert u.length == uDim;
		assert xn.length == xDim;

		if(x[0] >= 3) {
			return -3;
		} else if (x[0] == 2) {
			return -2;
		}
		return 0; // next to last state and absorbing state
	}

	/* (non-Javadoc)
	 * @see jrl.environment.Environment#startState()
	 */
	@Override
	public final double[] startState() {
		return new double[]{CHAIN_LENGTH-1};
	}

}
