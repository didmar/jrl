package com.github.didmar.jrl.environment;

import com.github.didmar.jrl.environment.dynsys.DynSysEnvironment;
import com.github.didmar.jrl.environment.dynsys.TwoDoFArmSystem;
import com.github.didmar.jrl.utils.array.ArrUtils;

/**
 * A reaching task based on {@link TwoDoFArmSystem}.
 * @author Didier Marin
 */
public final class TwoDoFArmEnvironment extends DynSysEnvironment {

	private final double[] xiTarget;
	private final double targetRadius;
	private final double targetMaxSpeed;
	private final double goalReward;
	private final double goalSpread;
	private final double costFactor;
	private final double outOfBoundsPenalty;
	private final double[] x0;
	private final boolean useArticularLimits;
	
	// used for temporary storage
	private final double[] q;
	private final double[] xi;
	private final double[] qd;
	private final double[] xid;

	public TwoDoFArmEnvironment(TwoDoFArmSystem dynSys, double dt,
			double[] xiTarget, double targetRadius, double targetMaxSpeed,
			double goalReward, double goalSpread, double costFactor,
		    double outOfBoundsPenalty, double[] qInit, double[] qMin, double[] qMax,
		    double[] qdMax) {
		super(dynSys, dt, computeXMin(qMin,qdMax), computeXMax(qMax,qdMax));
		if(qInit.length != TwoDoFArmSystem.NDOF) {
			throw new IllegalArgumentException("qInit must have length "+TwoDoFArmSystem.NDOF);
		}
		// Compute the operational target from the articular target
		this.xiTarget = xiTarget;
		this.targetRadius = targetRadius;
		this.targetMaxSpeed = targetMaxSpeed;
		this.goalReward = goalReward;
		this.goalSpread = goalSpread;
		this.costFactor = costFactor;
		this.outOfBoundsPenalty = outOfBoundsPenalty;
		x0 = ArrUtils.zeros(TwoDoFArmSystem.NDOF*2);
		System.arraycopy(qInit, 0, x0, 0, TwoDoFArmSystem.NDOF);
		
		useArticularLimits = true;
		
		q = new double[TwoDoFArmSystem.NDOF];
		xi = new double[TwoDoFArmSystem.NDOF];
		qd = new double[TwoDoFArmSystem.NDOF];
		xid = new double[TwoDoFArmSystem.NDOF];
		
	}
	
	/**
	 * Compute the lower bound of the state-space using the minimum articular
	 * position <code>qMin</code> and the maximum absolute articular speed
	 * <code>qdMax</code>.
	 * @param qMin	minimum articular position
	 * @param qdMax	maximum absolute articular speed
	 * @return	the lower bound of the state-space
	 */
	private static final double[] computeXMin(double[] qMin, double[] qdMax) {
		if(qMin.length != TwoDoFArmSystem.NDOF
				|| qdMax.length != TwoDoFArmSystem.NDOF) {
			throw new IllegalArgumentException("qMin and qdotMax must have "
					+"length "+TwoDoFArmSystem.NDOF);
		}
		for (int i = 0; i < qdMax.length; i++) {
			if(qdMax[i] < 0.) {
				throw new IllegalArgumentException("qdMax must be positive");
			}
		}
		double[] xMin = new double[TwoDoFArmSystem.NDOF*2]; 
		for (int i = 0; i < TwoDoFArmSystem.NDOF; i++) {
			xMin[i] = qMin[i];
			xMin[TwoDoFArmSystem.NDOF+i] = -qdMax[i];
		}
		return xMin;
	}
	
	/**
	 * Compute the upper bound of the state-space using the maximum articular
	 * position <code>qMax</code> and the maximum absolute articular speed
	 * <code>qdMax</code>.
	 * @param qMax	maximum articular position
	 * @param qdMax	maximum absolute articular speed
	 * @return	the upper bound of the state-space
	 */
	private static final double[] computeXMax(double[] qMax, double[] qdMax) {
		if(qMax.length != TwoDoFArmSystem.NDOF
				|| qdMax.length != TwoDoFArmSystem.NDOF) {
			throw new IllegalArgumentException("qMax and qdotMax must have "
				+"length "+TwoDoFArmSystem.NDOF);
		}
		for (int i = 0; i < qdMax.length; i++) {
			if(qdMax[i] < 0.) {
				throw new IllegalArgumentException("qdMax must be positive");
			}
		}
		double[] xMax = new double[TwoDoFArmSystem.NDOF*2]; 
		for (int i = 0; i < TwoDoFArmSystem.NDOF; i++) {
			xMax[i] = qMax[i];
			xMax[TwoDoFArmSystem.NDOF+i] = -qdMax[i];
		}
		return xMax;
	}

	/* (non-Javadoc)
	 * @see jrl.environment.Environment#startState()
	 */
	@Override
	public final double[] startState() {
		return x0;
	}

	/* (non-Javadoc)
	 * @see jrl.environment.Environment#reward(double[], double[], double[])
	 */
	@Override
	public final double reward(double[] x, double[] u, double[] xn) {
		assert x.length == xDim;
		assert u.length == uDim;
		assert xn.length == xDim;
		
		// Compute the cost of the articular acceleration
		double r = - costFactor * ArrUtils.norm(u);

		// If the reward is episodic
	    if(goalSpread <= 0.0) {
	        if( isGoalState(x) && isGoalState(xn) ) {
	            r += goalReward;
	        }
	    } else {
	        // Else, compute the gaussian reward function
	    	System.arraycopy(x, 0, q, 0, TwoDoFArmSystem.NDOF);
		    ((TwoDoFArmSystem)dynSys).qToXi(q, xi);
	        double targetDist = ArrUtils.euclideanDist(xi, xiTarget, TwoDoFArmSystem.NDOF);
	        r += goalReward * Math.exp( - targetDist / goalSpread );
	    }

	    // Check if out of workspace
	    if(useArticularLimits && hasReachedArticularLimits(q)) {
	        r += outOfBoundsPenalty;
	    }

	    return r;
	}

	/**
	 * Indicates if a given state is in the goal region.
	 * @param x	the state
	 * @return	true if <code>x</code> is in the goal region, false else
	 */
	private final boolean isGoalState(double[] x) {
		assert x.length == xDim;

	    // Check if target reached
		System.arraycopy(x, 0, q, 0, TwoDoFArmSystem.NDOF);
	    ((TwoDoFArmSystem)dynSys).qToXi(q, xi);
	    double dist = ArrUtils.euclideanDist(xi, xiTarget, TwoDoFArmSystem.NDOF);

	    if(dist <= targetRadius) {
	    	System.arraycopy(x, TwoDoFArmSystem.NDOF, qd, 0, TwoDoFArmSystem.NDOF);
	    	((TwoDoFArmSystem)dynSys).qToXi(qd, xid);
	        double speed = ArrUtils.norm( xid );
	        if(speed <= targetMaxSpeed) {
	            return true;
	        }
	    }
	    return false;
	}

	/* (non-Javadoc)
	 * @see jrl.environment.Environment#isTerminal(double[], double[], double[])
	 */
	@Override
	public final boolean isTerminal(double[] x, double[] u, double[] xn) {
		assert x.length == xDim;
		assert u.length == uDim;
		assert xn.length == xDim;
		
		// If the reward is episodic, check if target reached
	    if(goalSpread <= 0.0) {
	        // Both the current and the next state must within the goal region
	        // for the sample to be terminal
	        if(isGoalState(x) && isGoalState(xn)) {
	            return true;
	        }
	    }
	    // Check if out of workspace
	    System.arraycopy(xn, 0, q, 0, TwoDoFArmSystem.NDOF);
	    ((TwoDoFArmSystem)dynSys).qToXi(q, xi);
	    if(useArticularLimits && hasReachedArticularLimits(q)) {
	        return true;
	    }
	    return false;
	}

	/**
	 * Returns true is a given articular position is on or outside the
	 * state-space bounds, false else.
	 * @param q	the articular position
	 * @return	true is a given articular position is on or outside the
	 * 			state-space bounds, false else
	 */
	private final boolean hasReachedArticularLimits(double[] q) {
		assert q.length == TwoDoFArmSystem.NDOF;
		
		for (int i = 0; i < TwoDoFArmSystem.NDOF; i++) {
			if(q[i] <= xMin[i] || q[i] >= xMax[i]) {
				return true;
			}
		}
		return false;
	}
}
