package com.github.didmar.jrl.environment.acrobot;

import com.github.didmar.jrl.environment.Environment;
import com.github.didmar.jrl.utils.array.ArrUtils;

// TODO passer un système dynamique !
/**
 * A two-link underactuated robot must swing to raise its tip above a line
 * @author Didier Marin
 */
public final class Acrobot extends Environment {

	/** Time step of the simulation */
	public static final double dt = 0.02;
	/** Maximum absolute angular speed of the first link */
	public static final double maxDotTheta1 = 4.*Math.PI;
	/** Maximum absolute angular speed of the second link */
	public static final double maxDotTheta2 = 9.*Math.PI;
	/** Maximum torque for the second link */
	public static final double maxTau = 1.;
	/** Mass of the first link, in Kg */
	public static final double m1 = 1.;
	/** Mass of the second link, in Kg */
	public static final double m2 = 1.;
	/** Length of the first link, in m */
	public static final double l1 = 1.;
	/** Length of the first link, in m */
	public static final double l2 = 1.;
	/** Length to the center of mass of the first link, in m */
	public static final double lc1 = l1/2.;
	/** Length to the center of mass of the second link, in m */
	public static final double lc2 = l2/2.;
	/** Moment of inertia of the first link */
	public static final double I1 = 1.;
	/** Moment of inertia of the second link */
	public static final double I2 = 1.;
	/** Gravity */
	public static final double g = 9.81;
	/** Height of the bar */
	public final double goalHeight;
	/** State-space lower bound */
	private static final double[] xMin
		= new double[]{        0.,         0., -maxDotTheta1, -maxDotTheta2};
	/** State-space upper bound */
	private static final double[] xMax
		= new double[]{2.*Math.PI, 2.*Math.PI, +maxDotTheta1, +maxDotTheta2};
	/** Action-space lower bound */
	private static final double[] uMin = new double[]{-Acrobot.maxTau};
	/** Action-space upper bound */
	private static final double[] uMax = new double[]{+Acrobot.maxTau};

	/** Used to store the start state */
	protected final double[] x0;
	/** Used to store the next state */
	protected final double[] xn;


	/**
	 * Constructor.
	 * @param difficulty determines the height of the goal, must be in [0,1]
	 */
	public Acrobot(double difficulty) {
		super(4, 1);
		if(difficulty < 0. || difficulty > 1.) {
			throw new IllegalArgumentException("difficulty must be in [0,1]");
		}
		goalHeight = (difficulty*2.-1)*(l1+l2);
		x0 = ArrUtils.zeros(xDim);
		xn = new double[xDim];
	}

	/* (non-Javadoc)
	 * @see jrl.environment.Environment#startState()
	 */
	@Override
	public final double[] startState() {
		return x0;
	}

	/* (non-Javadoc)
	 * @see jrl.environment.Environment#nextState(double[], double[])
	 */
	@Override
	public final double[] nextState(double[] x, double[] u) {
		assert x.length == xDim;
		assert u.length == uDim;

		final double th1  = x[0];
		final double th2  = x[1];
		final double dTh1 = x[2];
		final double dTh2 = x[3];

		final double tau  = u[0];

		final double d1 =
				m1 * Math.pow(lc1,2) + m2 * (Math.pow(l1,2) + Math.pow(lc2,2)
				+ 2 * l1 * lc2 * Math.cos(th2) + I1 + I2);

		final double d2 =
				m2 * (Math.pow(lc2,2) + l1 * lc2 * Math.cos(th2)) + I2;

		final double phi1 =
				- m2 * l1 * lc2 * Math.pow(dTh2,2) * Math.sin(th2)
				- 2 * m2 * l1 * lc2 * dTh2 * dTh1 * Math.sin(th2);

		final double phi2 =
				m2 * lc2 * g * Math.cos(th1 + th2 - Math.PI/2);

		assert d1 != 0. : "Division by zero ahead";
		assert d1 - phi2 != 0. : "Division by zero ahead";
		assert (m2 * Math.pow(lc2,2) + I2 - Math.pow(d2,2) / d1) != 0.
				: "Division by zero ahead";

		final double ddTh2 =
				(tau + d2 * phi1 / d1 - phi2 )
				/ (m2 * Math.pow(lc2,2) + I2 - Math.pow(d2,2) / d1) ;

		final double ddTh1 =
				- (1./d1) * (d2 * ddTh2 + phi1);

		xn[0] = th1  + dt * dTh1;
		xn[1] = th2  + dt * dTh2;
		xn[2] = dTh1 + dt * ddTh1;
		xn[3] = dTh2 + dt * ddTh2;

		ArrUtils.boundVector(xn, xMin, xMax);

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

		// Tip below the bar ?
		if(Math.cos(x[0])*l1+Math.cos(x[0]+x[1])*l2 > -goalHeight) {
			return -1.; // Receive a penalty
		}
		return 0.;
	}

	/* (non-Javadoc)
	 * @see jrl.environment.Environment#isTerminal(double[], double[], double[])
	 */
	@Override
	public final boolean isTerminal(double[] x, double[] u, double[] xn) {
		assert x.length == xDim;
		assert u.length == uDim;
		assert xn.length == xDim;

		// Tip above the bar ?
		return (Math.cos(x[0])*l1+Math.cos(x[0]+x[1])*l2 < -goalHeight);
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
