package com.github.didmar.jrl.agent;

import com.github.didmar.jrl.environment.EnvironmentListener;
import com.github.didmar.jrl.policy.Policy;

/**
 * A Learning Agent is an agent that listen to its environment and update
 * its policy to maximize its expected reward.
 * 
 * @author Didier Marin
 */
public abstract class LearningAgent extends PolicyAgent implements EnvironmentListener {

	public LearningAgent(final Policy pol) {
		super(pol);
	}

}
