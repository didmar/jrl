package com.github.didmar.jrl.evaluation.vflearner.lstd;

import com.github.didmar.jrl.environment.EnvironmentListener;
import com.github.didmar.jrl.evaluation.valuefunction.LinearQFunction;
import com.github.didmar.jrl.evaluation.valuefunction.QFunction;
import com.github.didmar.jrl.evaluation.vflearner.QFunctionLearner;
import com.github.didmar.jrl.policy.Policy;
import com.github.didmar.jrl.utils.array.ArrUtils;

/**
 * Incremental Least-Square Temporal Difference learner of the state-action
 * value function.
 * @author Didier Marin
 */
public final class ILSTDQ implements QFunctionLearner, EnvironmentListener {

	/** State-action value function approximation */
	private final LinearQFunction qFunction;
	/** Number of state-action value function approximation parameters */
	private final int m;
	/** The policy we want to evaluate */
	private final Policy pol;
	/** Reward discount factor */
	private final double gamma;
	/** Eligibility factor */
	private final double lambda;
	/** Number of steps between each parameters update */
	private final int nbStepsBeforeUpdate;
	/** Number of steps before the next update */
	private int stepsBeforeUpdate;
	/** State-space dimension */
	private final int xDim;
	/** Action-space dimension */
	private final int uDim;
	
	private final double[][] Ainv;
	private final double[] b;
	private final double[] z;
	
	private final double[] xu;
	private final double[] xnun;
	private final double[] psixu; 
	private final double[] psixnun;
	private final double[] psixuMinusGammaPsixnun;
	private final double[] tmp1;
	private final double[][] tmp2;
	private final double[][] tmp3;
	
	public ILSTDQ(LinearQFunction qFunction, Policy pol, double gamma,
			double lambda, int nbStepsBeforeUpdate, double diagAinv0)
			throws Exception {
		if(gamma < 0. || gamma > 1.) {
			throw new IllegalArgumentException("gamma must be in [0,1]");
		}
		if(lambda < 0. || lambda > 1.) {
			throw new IllegalArgumentException("lambda must be in [0,1]");
		}
		if(diagAinv0 <= 0.) {
			throw new Exception("The initial value of the diagonal of the"
					+" statistics matrix must be positive");
		}
		this.qFunction = qFunction;
		this.pol = pol;
		this.gamma = gamma;
		this.lambda = lambda;
		this.nbStepsBeforeUpdate = nbStepsBeforeUpdate;
		stepsBeforeUpdate = this.nbStepsBeforeUpdate;
		m = qFunction.getParamsSize();
		// Initialize the inverted statistics matrix A
		Ainv = ArrUtils.eye(m,diagAinv0);
		b = ArrUtils.zeros(m);
		z = ArrUtils.zeros(m);
		xDim = qFunction.getXDim();
		uDim = qFunction.getUDim();
		
		xu   = new double[xDim+uDim];
		xnun = new double[xDim+uDim];
		psixu = new double[m];
		psixnun = new double[m];
		psixuMinusGammaPsixnun = new double[m];
		tmp1 = new double[m];
		tmp2 = new double[m][m];
        tmp3 = new double[m][m];
	}
	
	/* (non-Javadoc)
	 * @see jrl.environment.EnvironmentListener#newEpisode(double[], int)
	 */
	@Override
	public final void newEpisode(double[] x0, int maxT) {
		// Reinitialize eligibility traces vector
		ArrUtils.zeros(z);
	}

	/* (non-Javadoc)
	 * @see jrl.environment.EnvironmentListener#receiveSample(double[], double[], double[], double)
	 */
	@Override
	public final void receiveSample(double[] x, double[] u, double[] xn, double r, boolean isTerminal) {
		// Draw the next action
        pol.computePolicyDistribution(xn);
        final double[] un = pol.drawAction();
        // Compute the features
        System.arraycopy(x,  0,   xu,    0, xDim);
        System.arraycopy(u,  0,   xu, xDim, uDim);
        System.arraycopy(xn, 0, xnun,    0, xDim);
        System.arraycopy(un, 0, xnun, xDim, uDim);
        qFunction.getFeatures().phi(xu, psixu);
        qFunction.getFeatures().phi(xnun, psixnun);
        // Update the statistics
        for(int i=0; i<m; i++) {
        	z[i] = lambda*gamma*z[i] + psixu[i];
        	psixuMinusGammaPsixnun[i] = psixu[i] - gamma * psixnun[i];
        }
        ArrUtils.multiply(Ainv,z,tmp1,m,m);
        ArrUtils.multiply(tmp1,psixuMinusGammaPsixnun,tmp2,m);
        ArrUtils.multiply(tmp2, Ainv, tmp3, m, m, m);
        ArrUtils.multiplyQuad(psixuMinusGammaPsixnun, Ainv, tmp1, m);
        final double tmpScal = ArrUtils.dotProduct(tmp1, z, m); 
        for(int i=0; i<m; i++) {
        	for(int j=0; j<m; j++) {
        		Ainv[i][j] -= tmp3[i][j] / (1.+tmpScal);
        	}
        	b[i] += z[i]*r;
        }
        // Decrease the update counter, if it reaches zero compute the updated parameters
		stepsBeforeUpdate--;
		if(stepsBeforeUpdate == 0) {
			computeValueParameters();
			stepsBeforeUpdate = nbStepsBeforeUpdate;
		}

	}
	
	/* (non-Javadoc)
	 * @see jrl.environment.EnvironmentListener#endEpisode()
	 */
	@Override
	public final void endEpisode() {
		// Nothing to do
	}

	/**
	 * Update the value function approximation parameters based on the
	 * statistics.
	 */
	public final void computeValueParameters() {
		ArrUtils.multiply(Ainv, b, tmp1, m, m);
		qFunction.setParams(tmp1);
	}
	
	/* (non-Javadoc)
	 * @see jrl.evaluation.vflearner.QFunctionLearner#getQFunction()
	 */
	@Override
	public final QFunction getQFunction() {
		return qFunction;
	}
}
