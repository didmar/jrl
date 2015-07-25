package com.github.didmar.jrl.examples;

import com.github.didmar.jrl.agent.HumanAgent;
import com.github.didmar.jrl.environment.acrobot.Acrobot;
import com.github.didmar.jrl.environment.acrobot.AcrobotDisplay;

/**
 * Testing {@link HumanAgent}
 *
 * @author Didier Marin
 */
public class ExHumanAgent {

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		//---[ Create an acrobot environment ]----------------------------------
		final double difficulty = 0.25; // For Sutton's book setting, use 0.75
		final Acrobot env = new Acrobot(difficulty);
		final int uDim = env.getUDim();

		//---[ Create a display to visualize the test episodes ]----------------
		AcrobotDisplay disp = new AcrobotDisplay(env);
	    disp.openInJFrame(640, 480, "Acrobot test episodes");
	    env.addListener(disp);

		//---[ Create the human agent ]-----------------------------------------
		HumanAgent agent = new HumanAgent(uDim);

		env.interact(agent,1,1000);

	}

}
