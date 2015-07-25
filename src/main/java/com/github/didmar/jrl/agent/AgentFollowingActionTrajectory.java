package com.github.didmar.jrl.agent;

import org.eclipse.jdt.annotation.NonNull;

/**
 * An agent that uses a given sequence of actions, ignoring the state.
 * 
 * @author Didier Marin
 */
public final class AgentFollowingActionTrajectory implements Agent {
	
	/** Sequence of actions */
	private final double[][] us;
	/** Loop over the sequence ? (if not, the last action is repeated) */
	private final boolean loop;
	/** Index of the next action in the sequence */
	private int cpt;
	
	/**
	 * @param us	Sequence of actions
	 * @param loop	If True, loop the sequence, else,
	 *              the last action is repeated
	 */
	public AgentFollowingActionTrajectory(double[][] us, boolean loop) {
		if(us.length == 0) {
			throw new IllegalArgumentException("The action trajectory must contain at"
					+"least one action");
		}
		this.us = us;
		this.loop = loop;
		cpt = 0;
	}

	/* (non-Javadoc)
	 * @see jrl.agent.Agent#takeAction(double[])
	 */
	public final double[] takeAction(@NonNull final double[] x) {
		@SuppressWarnings("null")
		@NonNull final double[] u = us[cpt];
		cpt++;
		if(cpt == us.length) {
			if(loop) cpt = 0; else cpt--;
		}
		return u;
	}

}
