package com.github.didmar.jrl.environment;

import com.github.didmar.jrl.agent.Agent;
import com.github.didmar.jrl.environment.Environment;
import com.github.didmar.jrl.environment.EnvironmentListener;
import com.github.didmar.jrl.environment.IEnvironment;
import com.github.didmar.jrl.utils.array.ArrUtils;

/**
 * Decorator to make an environment partially observable. The state of the 
 * decorator correspond to the observation of the decorated environment.
 * The observation is computed through the computeObservation abstract method.
 * @author Didier Marin
 */
public abstract class PartiallyObservableEnvironment extends Environment {

	/** The fully observable environment */
	private final IEnvironment fullyObsEnv;
	/** Observation-space dimension */
	private final int oDim;
	
	// arrays for temporary storage to avoid mem. alloc.
	/** Used to store the observation */
	private final double[] o;
	/** Used to store the next observation */
	private final double[] on;
	
	public PartiallyObservableEnvironment(IEnvironment fullyObsEnv,	int oDim) {
		super(fullyObsEnv.getXDim(), fullyObsEnv.getUDim());
		if(oDim <= 0) {
			throw new IllegalArgumentException("Observation-space dimension must be greater than 0");
		}
		this.fullyObsEnv = fullyObsEnv;
		this.oDim = oDim;
		
		// init the arrays for temporary storage
		this.o = new double[this.oDim];
		this.on = new double[this.oDim];
	}

	/**
	 * Compute the observation of a given state.
	 * Precondition : x != null
	 * @param x the state
	 * @param o an array to store the observation
	 */
	public abstract void computeObservation(double[] x,	double[] o);
	
	/* (non-Javadoc)
	 * @see jrl.environment.Environment#startState()
	 */
	@Override
	public final double[] startState() {
		return fullyObsEnv.startState();
	}
	
	/**
	 * Returns the observed start state.
	 * @return the observed start state
	 */
	public final double[] startObservation() {
		computeObservation(fullyObsEnv.startState(), o);
		return o;
	}

	/* (non-Javadoc)
	 * @see jrl.environment.Environment#nextState(double[], double[])
	 */
	@Override
	public final double[] nextState(double[] x, double[] u) {
		return fullyObsEnv.nextState(x, u);
	}
	
	/**
	 * Returns the observed next state 
	 * @param x the state
	 * @param u the action
	 * @return the observed next state
	 */
	public final double[] nextObservation(double[] x, double[] u) {
		computeObservation(fullyObsEnv.nextState(x, u), o);
		return o;
	}

	/* (non-Javadoc)
	 * @see jrl.environment.Environment#reward(double[], double[], double[])
	 */
	@Override
	public final double reward(double[] x, double[] u, double[] xn) {
		return fullyObsEnv.reward(x, u, xn);
	}

	/* (non-Javadoc)
	 * @see jrl.environment.Environment#isTerminal(double[], double[], double[])
	 */
	@Override
	public final boolean isTerminal(double[] x, double[] u, double[] xn) {
		return fullyObsEnv.isTerminal(x, u, xn);
	}

	/* (non-Javadoc)
	 * @see jrl.environment.Environment#checkIfLegalAction(double[])
	 */
	@Override
	public final void checkIfLegalAction(double[] u) throws Exception {
		fullyObsEnv.checkIfLegalAction(u);
	}

	/* (non-Javadoc)
	 * @see jrl.environment.Environment#interact(jrl.agent.Agent, int, int, double[])
	 */
	@Override
	public final void interact(Agent agent, int nbEpi, int maxT, double[] x0) {
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
        	double[] x = (x0==null ? startState() : x0);
        	assert x.length == xDim : "Invalid initial state dimension : "+x.length+" instead of "+xDim;
        	assert getXMin()==null || ArrUtils.allGreaterOrEqual(x,getXMin())
        			: "Initial state is lower than inferior state bound";
        	assert getXMax()==null || ArrUtils.allLessOrEqual(x,getXMax())
        			: "Initial state is greater than superior state bound";
        	// Compute an observation of the initial state
        	computeObservation(x, o);
        	// Notify the beginning of a new episode to the fully observable
        	// environment listeners
        	for(EnvironmentListener l : fullyObsEnv) {
                l.newEpisode(x, maxT);
            }
        	// and also to the partially observable environment listeners,
        	// replacing the state by the observation
        	for(EnvironmentListener l : listeners) {
                l.newEpisode(o, maxT);
            }
            // Loop over decision steps
            for(int t=0; t<maxT; t++) {
            	// The agent takes an action based on the observation
            	final double[] u = agent.takeAction(o);
            	assert u.length == uDim : "Invalid action dimension";
                // Check that this action is legal
                try {
					checkIfLegalAction(u);
				} catch (Exception ex) {
					ex.printStackTrace();
				}
				// Draw the next state according to the current state and action
				final double[] xn = nextState(x,u);
				assert xn.length == xDim : "Invalid next state dimension : "+xn.length+" instead of "+xDim;
	        	assert getXMin()==null || ArrUtils.allGreaterOrEqual(xn,getXMin())
	        			: "Initial state is lower than inferior state bound";
	        	assert getXMax()==null || ArrUtils.allLessOrEqual(xn,getXMax())
	        			: "Initial state is greater than superior state bound";
                // Draw the reward according to the current state, the action
                // and the next state
                final double r = reward(x,u,xn);
                // Check whether or not the tuple (x,u,xn) is terminal 
                final boolean terminal = isTerminal(x,u,xn);
                // Compute the next observation
                computeObservation(xn, on);
                // Send the sample to the fully observable environment listeners
                for(EnvironmentListener l : fullyObsEnv) {
                    l.receiveSample(x, u, xn, r, terminal);
                }
                // and also to the partially observable environment listeners,
                // replacing the (next) state by the (next) observation
                for(EnvironmentListener l : listeners) {
                    l.receiveSample(o, u, on, r, terminal);
                }
                // If the sample is terminal, stop this episode
                if(terminal) {
                    break;
                }
                // Prepare the next step : next state becomes current state 
                x = xn;
                // and the next observation becomes the current observation
                System.arraycopy(on, 0, o, 0, oDim);
            }
            // Notify the end of the episode to both fully and partially
            // observable environment listeners
            for(EnvironmentListener l : fullyObsEnv) {
                l.endEpisode();
            }
            for(EnvironmentListener l : listeners) {
                l.endEpisode();
            }
        }
	}

	/* (non-Javadoc)
	 * @see jrl.environment.Environment#interact(jrl.agent.Agent, int, int)
	 */
	@Override
	public final void interact(Agent agent, int nbEpi, int maxT) {
		interact(agent, nbEpi, maxT, null);
	}

	/**
	 * Returns the observation dimension.
	 * @return the observation dimension
	 */
	public final int getODim() {
		return oDim;
	}
	
	
}
