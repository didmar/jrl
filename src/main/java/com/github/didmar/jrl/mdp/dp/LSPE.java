package com.github.didmar.jrl.mdp.dp;

import com.github.didmar.jrl.evaluation.valuefunction.LinearQFunction;
import com.github.didmar.jrl.mdp.DiscreteMDP;
import com.github.didmar.jrl.utils.DiscountFactor;
import com.github.didmar.jrl.utils.array.ArrUtils;

import Jama.Matrix;

/**
 * Least-Squares Policy Evaluation. See Lagoudakis and Parr 2003 "Least-squares
 * policy iteration.
 * @author Didier Marin
 */
public final class LSPE {

	/** The MDP to work with */
	private final DiscreteMDP mdp;
	/** Discount factor */
	private final DiscountFactor gamma;
	/** Q-Function represented as a matrix */
	private final LinearQFunction qFunction;
	/** The policy to evaluate */
	private final double[][] pol;
	/** Q-function features for each state-action */
	private final double[][] Phi;
	/** Reward for each state-action */
	private final double[][] R;

	private final double[][] Pi;

	private final double[][] P;

	public LSPE(DiscreteMDP mdp, double[][] pol, LinearQFunction qFunction,
			DiscountFactor gamma) {
		this.mdp = mdp;
		this.gamma = gamma;
		this.qFunction = qFunction;
		this.pol = pol;
		Phi = new double[mdp.n*mdp.m][qFunction.getParamsSize()];
		P = new double[mdp.n*mdp.m][mdp.n];
		Pi = new double[mdp.n][mdp.n*mdp.m];
		R = new double[mdp.n*mdp.m][1];
		// Perform Policy Evaluation
		performLSPE();
	}

	public final void performLSPE() {
		for(int x=0; x<mdp.n; x++) {
			for(int u=0; u<mdp.m; u++) {
				qFunction.getFeatures().phi(new double[]{x,u}, Phi[x+u*mdp.n]);
				R[x+mdp.n*u][0] = mdp.R[x][u];
				Pi[x][x+mdp.n*u] = pol[x][u];
				for(int xn=0; xn<mdp.n; xn++) {
					P[x+mdp.n*u][xn] = mdp.P[x][u][xn];
				}
			}
		}

		Matrix matPhi  = new Matrix(Phi);
		// A = Phi^T ( Phi - gamma P Pi Phi )
		Matrix A = matPhi.transpose().times(matPhi.minus(new Matrix(P).times(new Matrix(Pi)).times(matPhi).times(gamma.value)));
		// w = A^-1 b  with b = Phi^T R
		Matrix Ainv;
		try {
			Ainv = ArrUtils.pinv(A);
		} catch (Exception e) {
			// FIXME handle this exception properly
			throw new RuntimeException("Could not compute pseuso-inverse of A");
		}
		double[] w = Ainv.times(matPhi.transpose().times(new Matrix(R))).transpose().getArray()[0];
		qFunction.setParams(w);
	}
}
