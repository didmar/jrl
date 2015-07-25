package com.github.didmar.jrl.mdp;

import com.github.didmar.jrl.utils.RandUtils;
import com.github.didmar.jrl.utils.array.ArrUtils;

/**
 * Generic Average Reward Non-stationary Environment Testbed (GARNET) is an
 * arbitrary discrete MDP proposed in Bhatnagar et al. "Natural-Gradient
 * Actor-Critic algorithms". Non-stationarity is not implemented here !
 * 
 * @author Didier Marin
 */
public final class GARNETMDP extends DiscreteMDP {

	public GARNETMDP(int n, int m, int b) {
		super(n, m);
		ArrUtils.constvec(P0,1./((float)n)); // equiprobable start states
        // Compute the transition probabilities for each state
        for(int i=0; i<n; i++) {
        	createState(n,m,b,P[i],R[i]);
        }
	}
	
	/**
	 * Create a new GARNET state given parameters n, m and b, and store the
	 * its transition probabilities into Px and rewards into Rx.
	 * @param n    number of states
	 * @param m    number of actions
	 * @param b    number of possible next states given a state and an action
	 * @param Px   m-by-n array used to store the transition probabilities from
	 *             the created state for each action and next state
	 * @param Rx   m-by-n array used to store the rewards from the created state
	 *             for each action and next state
	 */
	public static final void createState(int n, int m, int b, double[][] Px,
			double[] Rx) {
		if(b < 1) {
			throw new IllegalArgumentException("b must be greater than 0");
		}
		// For each action
		for(int i=0; i<m; i++) {
			// Draw the reward from a normal law
			Rx[i] = RandUtils.nextGaussian(1.);
			if(b == 1) {
                ArrUtils.zeros(Px[i]);
                Px[i][RandUtils.nextInt(n)] = 1.;
			} else {
                // Shuffle indices between 0 and b-1, the b first elements will
                // be the possible next states
                final int[] ind = RandUtils.randPerm(n);
                // Compute the transition probabilities by partitioning
                // the unit interval at b-1 cut points
                final double[] cut = new double[b+1];
                cut[0] = 0.;
                for(int k=1; k<b; k++) {
                	cut[k] = RandUtils.nextDouble();
                }
                cut[b] = 1.;
                for(int k=1; k<b+1; k++) {
                	Px[i][ind[k-1]] = cut[k] - cut[k-1];	
                }
			}
		}
        
	}

}
