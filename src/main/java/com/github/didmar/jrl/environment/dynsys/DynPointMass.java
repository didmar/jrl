package com.github.didmar.jrl.environment.dynsys;

import com.github.didmar.jrl.environment.ReachingTask;
import com.github.didmar.jrl.utils.RandUtils;

/**
 * Reaching task with a point mass, with dynamics.
 * @author Didier Marin
 */
public final class DynPointMass extends DynSysEnvironment implements ReachingTask {

	/** Fixed start state. If null, the start state will be drawn uniformly
	 * between the state-space bounds */
	private double[] x0;
	private double targetPos;
	private final boolean goalIsTerminal;
	private double targetRadius;
	private double targetMaxSpeed;
	private final double goalReward;
	private final double costFactor;
	private final boolean randomStartState;
	
	public DynPointMass(double[] x0, double targetPos,
			double dt, double maxSpeed, boolean goalIsTerminal, double targetRadius,
			double targetMaxSpeed, double goalReward, double costFactor,
			boolean randomStartState) {
		super(new LinearDynanicSystem(new double[][]{{0.,1.},{0.,0.}},
				new double[][]{{0.},{1.}}), dt, new double[]{0.,-maxSpeed},
				new double[]{1.,+maxSpeed});
		if(x0.length != xDim) {
			throw new IllegalArgumentException("x0 must be of length xDim");
		}
		this.x0 = x0;
		setTargetPos(targetPos);
		this.goalIsTerminal = goalIsTerminal;
        this.targetRadius = targetRadius;
        this.targetMaxSpeed = targetMaxSpeed;
        this.goalReward = goalReward;
        this.costFactor = costFactor;
        this.randomStartState = randomStartState;
	}
	
	/* (non-Javadoc)
	 * @see jrl.environment.Environment#startState()
	 */
	@Override
	public final double[] startState() {
		if(randomStartState) {
			x0[0] = xMin[0] + RandUtils.nextDouble() * (xMax[0]-xMin[0]);
		}
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
	    double r = -costFactor*Math.pow(u[0], 2);
	    
        if(goalIsTerminal) {
            // If both the current and the next state are within the goal region
        	// we add a positive scalar reward and the episode will be over
            if( Math.abs(x[0]-targetPos) < targetRadius
            	&& Math.abs(x[1]) < targetMaxSpeed
                && Math.abs(xn[0]-targetPos) < targetRadius
                && Math.abs(xn[1]) < targetMaxSpeed ) {
                r += goalReward;
            }
        } else {
            // Gaussian reward centered on the target position
            double goalSpread = 0.01;
            r += goalReward * Math.exp(-Math.pow(xn[0]-targetPos,2)/goalSpread);
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
		
		// Target reached is the goal is considered terminal and both the
		// current and next state are within the goal region
		if( goalIsTerminal
			&& (Math.abs(x[0]-targetPos)  < targetRadius
			&&  Math.abs(x[1])            < targetMaxSpeed)
	        && (Math.abs(xn[0]-targetPos) < targetRadius
	        &&  Math.abs(xn[1])           < targetMaxSpeed) )
	    {
	        return true;
	    }
	    return false;
	}

	public void setTargetPos(double targetPos) {
		assert targetPos >= 0. && targetPos <= 1.;
		
		this.targetPos = targetPos;
	}
	
	public double getTargetRadius() {
		return targetRadius;
	}

	public void setTargetRadius(double targetRadius) {
		this.targetRadius = targetRadius;
	}

	public double getTargetMaxSpeed() {
		return targetMaxSpeed;
	}

	public void setTargetMaxSpeed(double targetMaxSpeed) {
		this.targetMaxSpeed = targetMaxSpeed;
	}
	
	@Override
	public double[] getTargetState(double[] x) {
		if(!goalIsTerminal) {
			throw new RuntimeException("The goal state must be terminal for the "
					+DynPointMass.class.getName()+" to be a "+ReachingTask.class.getName());
		}
		return new double[]{targetPos, 0.};
	}

	@Override
	public double[] getCurrentState(double[] x) {
		if(!goalIsTerminal) {
			throw new RuntimeException("The goal state must be terminal for the "
					+DynPointMass.class.getName()+" to be a "+ReachingTask.class.getName());
		}
		return x.clone();
	}

	@Override
	public double getCostFactor() {
		if(!goalIsTerminal) {
			throw new RuntimeException("The goal state must be terminal for the "
					+DynPointMass.class.getName()+" to be a "+ReachingTask.class.getName());
		}
		return costFactor;
	}

	@Override
	public double getGoalReward() {
		if(!goalIsTerminal) {
			throw new RuntimeException("The goal state must be terminal for the "
					+DynPointMass.class.getName()+" to be a "+ReachingTask.class.getName());
		}
		return goalReward;
	}

	public double getMaxSpeed() {
		return getXMax()[1];
	}

	public void setStartState(double[] x0) {
		this.x0 = x0;
	}
}
