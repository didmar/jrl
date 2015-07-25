package com.github.didmar.jrl.mdp;

/**
 * A simple MDP with two states and two actions. It was used in Kakade 2002
 * "A natural policy gradient" and then Bagnell & Schneider 2003 "Covariant
 * policy search" to illustrate the properties of natural policy gradients.
 * @author Didier Marin
 */
public final class TwoStateMDP extends DiscreteMDP {

	public TwoStateMDP() {
		super(new double[]{1.,0.},
			new double[][][]{{
				//x=0
					//xn=0  1
					    {1.,0.}, //u=0
				    	{0.,1.}  //u=1
				},{
				//x=1
					//xn=0  1
					    {0.,1.}, //u=0
					    {1.,0.}  //u=1
				}},
			new double[][]{
				//u=0  1
				   {1.,0.},   // x=0
				   {2.,0.}}); // x=1
	}

}
