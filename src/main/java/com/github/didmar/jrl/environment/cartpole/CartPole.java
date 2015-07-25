package com.github.didmar.jrl.environment.cartpole;

import com.github.didmar.jrl.environment.Environment;
import com.github.didmar.jrl.utils.RandUtils;

// TODO implementer comme un système dynamique !
/**
 * The cart-pole benchmark, as in Barto et al. 1983, but using the dynamic
 * without friction described in Florian 2007 "Correct equations for the
 * dynamics of the cart-pole system".
 * @author Didier Marin
 */
public final class CartPole extends Environment {

	public static final int POLE_POSITION = 0;
	public static final int POLE_SPEED    = 1;
	public static final int CART_POSITION = 2;
	public static final int CART_SPEED    = 3;

	public enum CartPoleRewardType {
		REWARD_IF_TARGET, EASY_REWARD, PUNISH_IF_NOT_TARGET;
	}

	/** Time step of the simulation */
	public static final double dt = 0.02;
	/** Length of the pole */
	public static final double l = 0.5;
	/** Mass of the pole */
	public static final double mp = 0.1;
	/** Mass of the cart */
	public static final double mc = 1.;
	/** Total mass */
	public static final double m = mc + mp;
	/** Gravity */
	public static final double g = 9.81;
	/** Maximum absolute force that can be applied to the cart */
	public static final double maxF = 10.;
	/** Maximum absolute cart position that is tolerated */
	public static final double maxPos = 2.4;
	/** Maximum absolute pole angle that is tolerated  */
	public static final double maxTh = 0.7;
	/** State-space lower bound */
	private static final double[] xMin
		= new double[]{-CartPole.maxTh, -5., -CartPole.maxPos, -2.};
	/** State-space upper bound */
	private static final double[] xMax
		= new double[]{+CartPole.maxTh, +5., +CartPole.maxPos, +2.};
	/** Action-space lower bound */
	private static final double[] uMin = new double[]{-CartPole.maxF};
	/** Action-space upper bound */
	private static final double[] uMax = new double[]{+CartPole.maxF};
	/** Maximum absolute cart position for which a positive reward is given */
	public static final double rewPos = 0.05;
	/** Maximum absolute pole angle for which a positive reward is given */
	public static final double rewTh = 0.05;
	/** Type of reward : */
	private CartPoleRewardType rewardType;
	/** If true, use a (not completely) random start state */
	private boolean randomStartState;
	/** Used if the start state is not random */
	protected final double[] defaultStartState;
	/** Used to store the start state */
	protected final double[] x0;
	/** Used to store the next state */
	protected final double[] xn;

	public CartPole(CartPoleRewardType rewardType, boolean randomStartState) {
		super(4, 1);
		this.rewardType = rewardType;
		this.randomStartState = randomStartState;
		defaultStartState = new double[xDim];
		defaultStartState[POLE_POSITION] = -0.2;
		defaultStartState[POLE_SPEED]    =  0.;
		defaultStartState[CART_POSITION] = +0.2;
		defaultStartState[CART_SPEED]    =  0.;
		x0 = defaultStartState.clone();
		xn = new double[xDim];
	}

	/* (non-Javadoc)
	 * @see jrl.environment.Environment#startState()
	 */
	@Override
	public final double[] startState() {
		if(randomStartState) {
			// Random pole angle (in [-0.2,0.2])
			x0[POLE_POSITION] = -0.2 + 0.4*RandUtils.nextDouble();
			// Random cart position (in [-0.5,+0.5])
			x0[CART_POSITION] = -0.5 + 1.0*RandUtils.nextDouble();
		}
		return x0;
	}

	/* (non-Javadoc)
	 * @see jrl.environment.Environment#nextState(double[], double[])
	 */
	@Override
	public final double[] nextState(double[] x, double[] u) {
		assert x.length == xDim;
		assert u.length == uDim;

		final double cosTh = Math.cos(x[POLE_POSITION]);
		final double sinTh = Math.sin(x[POLE_POSITION]);
		final double dotThSqrd = Math.pow(x[POLE_SPEED],2);
		final double F = Math.min(Math.max(u[0],-maxF),+maxF);
		// Compute the pole acceleration
		final double poleAccel =
			(g*sinTh + cosTh*(-F-mp*l*dotThSqrd*sinTh)/m)
			/ (l*(4./3 - mp*Math.pow(cosTh,2)/m));
		// Compute the cart acceleration
		final double cartAccel =
			(F + mp*l*(dotThSqrd*sinTh - poleAccel*cosTh)) / m;
		xn[POLE_POSITION] = x[POLE_POSITION] + dt * x[POLE_SPEED];
		xn[POLE_SPEED]    = x[POLE_SPEED]    + dt * poleAccel;
		xn[CART_POSITION] = x[CART_POSITION] + dt * x[CART_SPEED];
		xn[CART_SPEED]    = x[CART_SPEED]    + dt * cartAccel;
		return xn;
	}

	/* (non-Javadoc)
	 * @see jrl.environment.Environment#reward(double[], double[], double[])
	 */
	@Override
	public final double reward(double[] x, double[] u, double[] xn) {
		assert x.length == xDim;
		assert u.length == uDim;
		assert xn.length == xDim;

		double r = 0.;
		switch(rewardType) {
			case REWARD_IF_TARGET :
				r = (Math.abs(x[POLE_POSITION]) < rewTh
						|| Math.abs(x[CART_POSITION]) < rewPos) ? 1. : 0.;
				break;
			case EASY_REWARD :
				r = - Math.abs(x[POLE_POSITION]);
				break;
			case PUNISH_IF_NOT_TARGET :
				if (Math.abs(x[POLE_POSITION]) < rewTh
						|| Math.abs(x[CART_POSITION]) < rewPos) {
					r = 0.;
				} else if(Math.abs(x[POLE_POSITION]) > maxTh
						|| Math.abs(x[CART_POSITION]) > maxPos) {
					r = -100.; // Big punishment for going out of bounds
				} else {
					return -1.; // Smaller punishment for not being in target region
				}
		}
		return r;
	}

	/* (non-Javadoc)
	 * @see jrl.environment.Environment#isTerminal(double[], double[], double[])
	 */
	@Override
	public final boolean isTerminal(double[] x, double[] u, double[] xn) {
		assert x.length == xDim;
		assert u.length == uDim;
		assert xn.length == xDim;

		return (Math.abs(x[POLE_POSITION]) > maxTh
				|| Math.abs(x[CART_POSITION]) > maxPos);
	}

	/**
	 * Switch between random and fixed start state
	 * @param use switch to random start state if true, fixed start state else
	 */
	public final void useRandomStartState(boolean use) {
		randomStartState = use;
		if(!randomStartState) {
			System.arraycopy(defaultStartState, 0, x0, 0, xDim);
		}
	}

	/* (non-Javadoc)
	 * @see jrl.environment.Environment#getXMin()
	 */
	@Override
	public final double[] getXMin() {
		return xMin;
	}

	/* (non-Javadoc)
	 * @see jrl.environment.Environment#getXMax()
	 */
	@Override
	public final double[] getXMax() {
		return xMax;
	}

	/* (non-Javadoc)
	 * @see jrl.environment.Environment#getUMin()
	 */
	@Override
	public final double[] getUMin() {
		return uMin;
	}

	/* (non-Javadoc)
	 * @see jrl.environment.Environment#getUMax()
	 */
	@Override
	public final double[] getUMax() {
		return uMax;
	}
}
