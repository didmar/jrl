package com.github.didmar.jrl.environment;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import org.eclipse.jdt.annotation.NonNull;
import org.eclipse.jdt.annotation.Nullable;

import com.github.didmar.jrl.agent.Agent;
import com.github.didmar.jrl.utils.array.ArrUtils;

/**
 * Partial implementation of the {@link IEnvironment} interface. For a complete
 * implementation, see {@link PointMass} for instance.
 * @author Didier Marin
 */
public abstract class Environment implements IEnvironment {

	/** State-space dimension */
	protected final int xDim;
	/** Action-space dimension */
	protected final int uDim;
	/** List of the listeners that are listening to the environment */
	protected final List<EnvironmentListener> listeners;

	public Environment(int xDim, int uDim) {
		if(xDim <= 0) {
			throw new IllegalArgumentException("State-space dimension must be greater than 0");
		}
		if(uDim <= 0) {
			throw new IllegalArgumentException("Action-space dimension must be greater than 0");
		}
        this.xDim = xDim;
        this.uDim = uDim;
        listeners = new ArrayList<EnvironmentListener>();
	}

    /* (non-Javadoc)
	 * @see jrl.environment.IEnvironment#startState()
	 */
	public abstract double[] startState();

    /* (non-Javadoc)
	 * @see jrl.environment.IEnvironment#nextState(double[], double[])
	 */
	public abstract double[] nextState(double[] x, double[] u);

    /* (non-Javadoc)
	 * @see jrl.environment.IEnvironment#reward(double[], double[], double[])
	 */
	public abstract double reward(double[] x, double[] u, double[] xn);

    /* (non-Javadoc)
	 * @see jrl.environment.IEnvironment#isTerminal(double[], double[], double[])
	 */
	public abstract boolean isTerminal(double[] x, double[] u, double[] xn);


    /**
     * An implementation that checks for null reference and NaN values.
     * Should be overridden to incorporate environment-specific constraints.
	 * @see com.github.didmar.jrl.environment.IEnvironment#checkIfLegalAction(double[])
	 */
	public void checkIfLegalAction(double[] u) throws Exception {
        for(double ui : u) {
        	if(Double.isNaN(ui)) {
        		throw new Exception("Illegal action : NaN");
        	}
        }
        @Nullable double[] uMin = getUMin();
    	if(uMin != null) {
    		if(!ArrUtils.allGreaterOrEqual(u, uMin)) {
    			throw new Exception("Illegal action : less than lower bound");
    		}
    	}
    	@Nullable double[] uMax = getUMax();
    	if(uMax != null) {
    		if(!ArrUtils.allLessOrEqual(u, uMax)) {
    			throw new Exception("Illegal action : greater than upper bound");
    		}
    	}
    }

    /* (non-Javadoc)
	 * @see jrl.environment.IEnvironment#interact(jrl.agent.Agent, int, int, double[])
	 */
	public void interact(Agent agent, int nbEpi, int maxT, @Nullable double[] x0) {
		if(nbEpi < 0) {
			throw new IllegalArgumentException("nbEpi must be positive");
		}
		if(maxT <= 0) {
			throw new IllegalArgumentException("nbEpi must be greater than 0");
		}
		if(x0 != null && x0.length != xDim) {
			throw new IllegalArgumentException("Given initial state x0 does not"
					+"match the state-space dimension");
		}
        // Loop over episodes
        for(int e=0; e<nbEpi; e++) {
            // If no start state was specified, draw one
        	@NonNull double[] x = (x0==null ? startState() : x0);
        	assert x.length == xDim : "Invalid initial state dimension : "+x.length+" instead of "+xDim;
        	assert stateAboveLowerBound(x) : "Initial state is lower than inferior state bound";
        	assert stateBelowUpperBound(x) : "Initial state is greater than superior state bound";
        	// Notify the beginning of a new episode to the listeners
            for(EnvironmentListener l : listeners) {
                l.newEpisode(x, maxT);
            }
            // Loop over decision steps
            for(int t=0; t<maxT; t++) {
            	// The agent takes an action
            	@NonNull final double[] u = agent.takeAction(x);
            	assert u.length == uDim : "Invalid action dimension";
                // Check that this action is legal
                try {
					checkIfLegalAction(u);
				} catch (Exception ex) {
					ex.printStackTrace();
				}
				// Draw the next state according to the current state and action
				@NonNull final double[] xn = nextState(x,u);

				assert xn.length == xDim : "Invalid next state dimension : "+xn.length+" instead of "+xDim;
				assert stateAboveLowerBound(xn) : "Next state is lower than inferior state bound";
	        	assert stateBelowUpperBound(xn) : "Next state is greater than superior state bound";

                // Draw the reward according to the current state, the action
                // and the next state
                final double r = reward(x,u,xn);
                // Check whether the tuple (x,u,xn) is terminal or not
                final boolean terminal = isTerminal(x,u,xn);
                // Send the sample to the listeners
                for(EnvironmentListener l : listeners) {
                    l.receiveSample(x, u, xn, r, terminal);
                }
                // If the sample is terminal, stop this episode
                if(terminal) {
                    break;
                }
                // Prepare the next step : next state becomes current state
                x = xn;
            }
            // Notify the end of the episode to the listeners
            for(EnvironmentListener l : listeners) {
                l.endEpisode();
            }
        }
	}

    /* (non-Javadoc)
	 * @see jrl.environment.IEnvironment#interact(jrl.agent.Agent, int, int)
	 */
    public void interact(Agent agent, int nbEpi, int maxT) {
    	interact(agent, nbEpi, maxT, null);
	}

	/* (non-Javadoc)
	 * @see jrl.environment.IEnvironment#addListener(jrl.environment.EnvironmentListener)
	 */
	public final void addListener(EnvironmentListener listener) {
        listeners.add(listener);
    }

    /* (non-Javadoc)
	 * @see jrl.environment.IEnvironment#removeListener(jrl.environment.EnvironmentListener)
	 */
	public final boolean removeListener(EnvironmentListener listener) {
        return listeners.remove(listener);
    }

	/* (non-Javadoc)
     * @see jrl.environment.IEnvironment#removeAllListener()
     */
    public final void removeAllListener() {
    	listeners.clear();
	}

	/* (non-Javadoc)
	 * @see jrl.environment.IEnvironment#getXDim()
	 */
	public final int getXDim() {
		return xDim;
	}

    /* (non-Javadoc)
	 * @see jrl.environment.IEnvironment#getUDim()
	 */
	public final int getUDim() {
		return uDim;
	}

	/* (non-Javadoc)
	 * @see jrl_testing.environment.IEnvironment#getXMin()
	 */
	@Nullable
	public double[] getXMin() {
		// No lower state bound by default
		return null;
	}

	/* (non-Javadoc)
	 * @see jrl_testing.environment.IEnvironment#getXMax()
	 */
	@Nullable
	public double[] getXMax() {
		// No upper state bound by default
		return null;
	}

	/* (non-Javadoc)
	 * @see jrl_testing.environment.IEnvironment#getUMin()
	 */
	@Nullable
	public double[] getUMin() {
		// No lower action bound by default
		return null;
	}

	/* (non-Javadoc)
	 * @see jrl_testing.environment.IEnvironment#getUMax()
	 */
	@Nullable
	public double[] getUMax() {
		// No upper action bound by default
		return null;
	}

	/**
	 * Iterate over the environment listeners
	 * @see java.lang.Iterable#iterator()
	 */
	@Nullable
	public final Iterator<EnvironmentListener> iterator() {
		return listeners.iterator();
	}

	private final boolean stateAboveLowerBound(double[] x) {
    	@Nullable final double[] xmin = getXMin();
    	if(xmin==null) return false;
    	return ArrUtils.allGreaterOrEqual(x,xmin);
	}

    private final boolean stateBelowUpperBound(double[] x) {
    	@Nullable final double[] xmax = getXMax();
    	if(xmax==null) return false;
    	return ArrUtils.allLessOrEqual(x,xmax);
	}
}
