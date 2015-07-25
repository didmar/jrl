package com.github.didmar.jrl.environment.dynsys;

import com.github.didmar.jrl.utils.Utils;
import com.github.didmar.jrl.utils.array.ArrUtils;

/**
 * A robotic arm with two degrees of freedom. The state space is x = [q,qd]
 * with q the articular position and qd the articular speed. The action u is
 * a torque applied to the arm. An external constant force can be added.
 * @author Didier Marin
 */
public final class TwoDoFArmSystem implements DynamicSystem {

	public static final int NDOF = 2;

	private final double l1;
	private final double l2;
	private final double m1;
	private final double m2;
	private final double armStrength;
	private final double[] forceField;

	// used for temporary storage
	double[] q = new double[NDOF];
	double[] qd = new double[NDOF];
	double[] qdd = new double[NDOF];
	double[][] M = new double[NDOF][NDOF];
	double[][] Minv = new double[NDOF][NDOF];
	double[] C = new double[NDOF];
	double[] Fext = new double[NDOF];
	double[][] J = new double[NDOF][NDOF];
	double[] F = new double[NDOF];

	public TwoDoFArmSystem(double l1, double l2, double m1, double m2,
			double armStrength, double[] forceField) {
		this.l1 = l1;
		this.l2 = l2;
		this.m1 = m1;
		this.m2 = m2;
		this.armStrength = armStrength;
		this.forceField = forceField;
	}

	/* (non-Javadoc)
	 * @see jrl.environment.dynsys.DynamicSystem#dotX(double[], double[], double[])
	 */
	@Override
	public final void dotX(double[] x, double[] u, double[] dotX) {
		assert(x.length == NDOF*2);
		assert(u.length == NDOF);
		assert(dotX.length == NDOF*2);

	    System.arraycopy(x, 0,     q, 0, NDOF);
	    System.arraycopy(x, NDOF, qd, 0, NDOF);

	    // Compute the articular acceleration at the begining of the step
	    acceleration(q, qd, u, qdd);

	    System.arraycopy(qd,  0, dotX, 0,    NDOF);
	    System.arraycopy(qdd, 0, dotX, NDOF, NDOF);
	}

	public final void acceleration(double[] q, double[] qd, double[] tau,
			double[] qdd) {
		assert q.length == NDOF;
		assert qd.length == NDOF;
		assert tau.length == NDOF;
		assert qdd.length >= NDOF;

	    // Compute the inertia matrix and invert it
	    inertiaMatrix(q, M);
	    try {
			ArrUtils.invert2x2(M, Minv);
		} catch (Exception e) {
			// FIXME handle this exception properly
			throw new RuntimeException("Could not invert inertia matrix");
		}

	    // Compute the Coriolis matrix
	    coriolisVector(q, qd, C);

	    // Compute the external forces
	    if(forceField != null) {
		    kinematicJacobian(q, J);
		    ArrUtils.multiply(J, forceField, Fext, NDOF, NDOF);
	    } else {
	    	ArrUtils.zeros(Fext);
	    }

	    // Compute the articular acceleration
	    for (int i = 0; i < NDOF; i++) {
			F[i] = tau[i] * armStrength - C[i] + Fext[i];
		}
	    ArrUtils.multiply(Minv, F, qdd, NDOF, NDOF);

	}

	public final void kinematicJacobian(double[] q, double[][] J) {
		assert q.length == NDOF;
		assert ArrUtils.hasShape(J, NDOF, NDOF);

	    J[0][0] = - Math.sin(q[0])*l1 - Math.sin(q[0]+q[1])*l2;
	    J[1][0] = Math.cos(q[0])*l1 + Math.cos(q[0]+q[1])*l2;
	    J[0][1] = - Math.sin(q[0]+q[1])*l2;
	    J[1][1] = Math.cos(q[0]+q[1])*l2;
	}

	public final void qToXi(double[] q, double[] xi) {
		assert q.length == NDOF;
		assert xi.length == NDOF;

		kinematicJacobian(q, J);
		ArrUtils.multiply(J, q, xi, NDOF, NDOF);
	}

	public final boolean inverseKinematicModel(double[] xi, double[] q) {
		assert xi.length == NDOF;
		assert q.length == NDOF;

	    if(!isPositionReachable(xi)) {
	        return false;
	    }

	    if(ArrUtils.norm(xi) == l1+l2)
	    {
	        q[0] = Utils.boundAngle(Math.acos(xi[0]/(l1+l2)));
	        q[1] = 0.0;
	    } else {

		    // Compute the position of the arm-forearm joint (one of the two possibles)
		    double x;
		    double y;
		    if(xi[1]==0)
		    {
		        x = (Math.pow(l2,2)-Math.pow(l1,2)-Math.pow(xi[0],2)) / (-2*xi[0]);
		        double a = 1;
		        double b = -2*xi[1];
		        double c = Math.pow(xi[0],2)+Math.pow(x,2)-2*xi[0]*x+Math.pow(xi[1],2)-Math.pow(l2,2);
		        double delta = Math.sqrt( Math.pow(b,2) - 4*a*c );
		        y = (-b+delta)/(2*a);
		    } else {
		    	double N = (Math.pow(l2,2)-Math.pow(l1,2)-Math.pow(xi[0],2)-Math.pow(xi[1],2))/(-2*xi[1]);
		    	double a = Math.pow((xi[0]/xi[1]),2)+1;
		    	double b = -2*N*(xi[0]/xi[1]);
		    	double c = Math.pow(N,2)-Math.pow(l1,2);
		    	double delta = Math.sqrt( Math.pow(b,2) - 4*a*c );
		        // Compute the position of the arm-forearm joint
		        x = (-b+delta)/(2*a);
		        y = (Math.pow(l2,2)-Math.pow(l1,2)-Math.pow(xi[0],2)-Math.pow(xi[1],2))/(-2*xi[1]) - x*(xi[0]/xi[1]);
		    }

		    // Compute the corresponding articular position
		    q[0] = y > 0.0 ? Math.acos(x/l1) : -Math.acos(x/l1);
		    q[1] = (xi[1]-y>0.0 ?
		    		Math.acos((xi[0]-x)/l2) : -Math.acos((xi[0]-x)/l2)) -q[0];
	    }
	    Utils.boundAngles(q);
	    return true;
	}

	public final boolean isPositionReachable(double[] xi) {
		assert xi.length == NDOF;

		return ArrUtils.norm(xi) <= l1+l2;
	}

	public final void coriolisVector(double[] q, double[] qd, double[] C) {
		assert q.length == NDOF;
		assert qd.length == NDOF;

		final double m2_l1_l2_sinq2 = m2*l1*l2*Math.sin(q[1]);

	    C[0] = -qd[1]*(2*qd[0]+qd[1])*m2_l1_l2_sinq2;
	    C[1] = Math.pow(qd[0],2)*m2_l1_l2_sinq2;
	}

	public final void inertiaMatrix(double[] q, double[][] M) {
		assert q.length == NDOF;
		assert ArrUtils.hasShape(M, NDOF, NDOF);

	    final double l2squared_m2 = Math.pow(l2,2)*m2;
	    final double l1_l2_m2_cosq2 = l1*l2*m2*Math.cos(q[1]);

	    M[0][0] = l2squared_m2 + 2*l1_l2_m2_cosq2 + Math.pow(l1,2)*(m1+m2);
	    M[0][1] = l2squared_m2 + l1_l2_m2_cosq2;
	    M[1][0] = M[0][1];
	    M[1][1] = l2squared_m2;
	}

	/* (non-Javadoc)
	 * @see jrl.environment.dynsys.DynamicSystem#derX(double[], double[], double[][])
	 */
	@Override
	public final void derX(double[] x, double[] u, double[][] derX) {
		throw new UnsupportedOperationException();

	}

	/* (non-Javadoc)
	 * @see jrl.environment.dynsys.DynamicSystem#derU(double[], double[], double[][])
	 */
	@Override
	public final void derU(double[] x, double[] u, double[][] derU) {
		throw new UnsupportedOperationException();

	}

	/* (non-Javadoc)
	 * @see jrl.environment.dynsys.DynamicSystem#getXDim()
	 */
	@Override
	public final int getXDim() {
		return NDOF*2;
	}

	/* (non-Javadoc)
	 * @see jrl.environment.dynsys.DynamicSystem#getUDim()
	 */
	@Override
	public final int getUDim() {
		return NDOF;
	}

}
