package com.github.didmar.jrl.environment.dynsys;

import com.github.didmar.jrl.environment.Environment;
import com.github.didmar.jrl.utils.array.ArrUtils;
import com.github.didmar.jrl.utils.rk4.RkRoutine;
import com.github.didmar.jrl.utils.rk4.RungeKutta4;

// TODO choix d'une autre méthode d'intégration (Runge-Kutta)
/**
 * An {@link Environment} based on a {@link DynamicSystem}.
 * @author Didier Marin
 */
public abstract class DynSysEnvironment extends Environment {

	protected final DynamicSystem dynSys;
	/** Simulation time-step */
	public final double dt;
	/** State-space lower bound */
	protected final double[] xMin;
	/** State-space upper bound */
	protected final double[] xMax;

	private final RungeKutta4<double[]> rk4 = new RungeKutta4<double[]>();

	/** Used to store the next state */
	private final double[] xn;
	/** Used to store the state-space speed */
	private final double[] dX;

	public DynSysEnvironment(DynamicSystem dynSys, double dt,
			double[] xMin, double[] xMax) {
		super(dynSys.getXDim(), dynSys.getUDim());
		if(dt <= 0.) {
			throw new IllegalArgumentException("dt must be greater than 0");
		}
		if(xMin != null && xMin.length != xDim) {
			throw new IllegalArgumentException("xMin must have length xDim (or be null)");
		}
		if(xMax != null && xMax.length != xDim) {
			throw new IllegalArgumentException("xMax must have length xDim (or be null)");
		}
		this.dynSys = dynSys;
		this.dt = dt;
		this.xMin = xMin;
		this.xMax = xMax;

		xn = new double[xDim];
		dX = new double[xDim];
	}

	/* (non-Javadoc)
	 * @see jrl.environment.Environment#nextState(double[], double[])
	 */
	@Override
	public final double[] nextState(double[] x, double[] u) {
		assert x.length == xDim;
		assert u.length == uDim;

		// RK4 method
		final RkDotXRoutine rkRoutine = new RkDotXRoutine(u);
		System.arraycopy(rk4.rk4(x, 0., dt, 1, rkRoutine), 0, xn, 0, xDim);

		// Euler method
//		dynSys.dotX(x, u, dX);
//		for (int i = 0; i < xDim; i++) {
//			xn[i] = x[i] + dt * dX[i];
//		}

		ArrUtils.boundVector(xn,xMin,xMax);

        return xn;
	}

	public class RkDotXRoutine implements RkRoutine<double[]> {

		private final double[] dotX;
		private final double[] u;

		public RkDotXRoutine(double[] u) {
			this.u = u;
			dotX = new double[xDim];
		}

		@Override
		public double[] rkRoutine(double[] X, double t, double dt) {
			dynSys.dotX(X, u, dotX);
			return dotX;
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

	public final DynamicSystem getDynSys() {
		return dynSys;
	}
}
