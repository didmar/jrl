package com.github.didmar.jrl.agent;

import javax.swing.JOptionPane;

import org.eclipse.jdt.annotation.NonNull;

import com.github.didmar.jrl.utils.array.ArrUtils;

/**
 * An agent that ask the user to type each action. Each dimension must be
 * separated by a comma.
 * 
 * @author Didier Marin
 */
public final class HumanAgent implements Agent {

	/** Action-space dimension */ 
	private final int uDim;

	public HumanAgent(int uDim) {
		this.uDim = uDim;
	}
	
	/* (non-Javadoc)
	 * @see jrl_testing.agent.Agent#takeAction(double[])
	 */
	@NonNull
	public final double[] takeAction(@NonNull final double[] x) {
		boolean good = true;
		String[] tokens = null;
		double[] u = new double[uDim];
		do {
			do {
				String uString = null;
				do {
					uString = JOptionPane.showInputDialog(dialogMessage(x));
				} while(uString == null);
				tokens = uString.split(",");
			} while(tokens.length != uDim);
			try {
				for(int i=0; i<tokens.length; i++) {
					u[i] = Double.parseDouble(tokens[i]);
				}
			} catch (NumberFormatException e) {
				good = false;
			}
		} while(!good);
		return u;
	}
	
	protected final static String dialogMessage(final double[] x) {
		return "x="+ArrUtils.toString(x)+"\nu ?";
	}
	
}
