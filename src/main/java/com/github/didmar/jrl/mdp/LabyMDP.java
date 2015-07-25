package com.github.didmar.jrl.mdp;

import com.github.didmar.jrl.utils.array.ArrUtils;

// TODO a tester
/**
 * MDP representing a 2D grid with obstacles and reward sources.
 * @author Didier Marin
 */
public final class LabyMDP extends DiscreteMDP {

	public static final int NB_ACTIONS = 5;
	public static final int NORTH = 0, SOUTH = 1, WEST = 2, EAST = 3, IDLE = 4;
	
	public final int width;
	public final int height;
	
	public LabyMDP(int width, int height) {
		super(ArrUtils.constvec(width*height, 1./((double)(width*height))),
				buildGridTransitions(width, height),
				ArrUtils.zeros(width*height,NB_ACTIONS));
		this.width = width;
		this.height = height;
	}

	private static final double[][][] buildGridTransitions(int width, int height) {
		double[][][] P = new double[width*height][NB_ACTIONS][width*height];
		for (int i = 0; i < width; i++) {
			for (int j = 0; j < height; j++) {
				int x = j*width + i;
				ArrUtils.zeros(P[x]);
				// North
				if(j > 0) {
					P[x][NORTH][x-width] = 1.;
				} else {
					P[x][NORTH][x] = 1.;
				}
				// South
				if(j < height-1) {
					P[x][SOUTH][x+width] = 1.;
				} else {
					P[x][SOUTH][x] = 1.;
				}
				// West
				if(i > 0) {
					P[x][WEST][x-1] = 1.;
				} else {
					P[x][WEST][x] = 1.;
				}
				// East
				if(i < width-1) {
					P[x][EAST][x+1] = 1.;
				} else {
					P[x][EAST][x] = 1.;
				}
				// Idle
				P[x][IDLE][x] = 1.;
			}
		}
		assert(verifyP(P));
		return P;
	}
	
	public final void setReward(int xcoord, int ycoord, double r) {
		R[ycoord*width+xcoord][IDLE] = r;
	}
	
	public final void setObstacle(int xcoord, int ycoord) {
		int x = ycoord*width+xcoord;
		ArrUtils.zeros(P[x]);
		for (int u = 0; u < NB_ACTIONS; u++) {
			P[x][u][x] = 1.;
		}
		// from North 
		if(ycoord > 0) {
			ArrUtils.zeros(P[x-width][SOUTH]);
			P[x-width][SOUTH][x-width] = 1.;
		}
		// from South
		if(ycoord < height-1) {
			ArrUtils.zeros(P[x+width][NORTH]);
			P[x+width][NORTH][x+width] = 1.;
		}
		// from West
		if(xcoord > 0) {
			ArrUtils.zeros(P[x-1][EAST]);
			P[x-1][EAST][x-1] = 1.;
		}
		// from East
		if(xcoord < width-1) {
			ArrUtils.zeros(P[x+1][WEST]);
			P[x+1][WEST][x+1] = 1.;
		}
		// Recompute P0
		int nbObstacles = 0;
		for (int i = 0; i < width; i++) {
			for (int j = 0; j < height; j++) {
				if(isObstacle(i, j)) {
					nbObstacles++;
				}
			}
		}
		for (int i = 0; i < width; i++) {
			for (int j = 0; j < height; j++) {
				if(isObstacle(i, j)) {
					P0[ycoord*width+xcoord] = 0.;
				} else {
					P0[ycoord*width+xcoord] = 1./((double)nbObstacles);
				}
			}
		}
	}
	
	public final boolean isObstacle(int xcoord, int ycoord) {
		int x = ycoord*width+xcoord;
		for (int u = 0; u < NB_ACTIONS; u++) {
			if(P[x][u][x] != 1.) {
				return false;
			}
		}
		// from North 
		if(ycoord > 0) {
			if(P[x-width][SOUTH][x-width] != 1.) {
				return false;
			}
		}
		// from South
		if(ycoord < height-1) {
			if(P[x+width][NORTH][x+width] != 1.) {
				return false;
			}
		}
		// from West
		if(xcoord > 0) {
			if(P[x-1][EAST][x-1] != 1.) {
				return false;
			}
		}
		// from East
		if(xcoord < width-1) {
			if(P[x+1][WEST][x+1] != 1.) {
				return false;
			}
		}
		return true;
	}
	
	public final void printLaby() {
		for (int j = 0; j < height; j++) {
			for (int i = 0; i < width; i++) {
				int x = j*width+i;
				if(isObstacle(i, j)) {
					System.out.print("#");
				} else {
					double r = ArrUtils.sum(R[x]);
					if(r > 0.) {
						System.out.print("+");
					} else if(r < 0.) {
						System.out.print("-");
					} else {
						System.out.print(".");
					}
				}
			}
			System.out.println();
		}
	}
	
	public final void printPolicy(double[][] probas) {
		for (int j = 0; j < height; j++) {
			for (int i = 0; i < width; i++) {
				if(isObstacle(i, j)) {
					System.out.print("#");
				} else {
					int x = j*width+i;
					boolean allEquals = true;
					for (int k = 1; k < probas[x].length; k++) {
						if(probas[x][0] != probas[x][k]) {
							allEquals = false;
							break;
						}
					}
					if(allEquals) {
						System.out.print("*");
					} else {
						int u = ArrUtils.argmax(probas[x]);
						switch(u) {
							case NORTH: System.out.print("^"); break;
							case SOUTH: System.out.print("v"); break;
							case WEST:  System.out.print("<"); break;
							case EAST:  System.out.print(">"); break;
							case IDLE:  System.out.print("."); break;
						}
					}
				}
			}
			System.out.println();
		}
	}
	
	public final void printPolicy(int[] pol) {
		for (int j = 0; j < height; j++) {
			for (int i = 0; i < width; i++) {
				if(isObstacle(i, j)) {
					System.out.print("#");
				} else {
					int x = j*width+i;
					switch(pol[x]) {
						case NORTH: System.out.print("^"); break;
						case SOUTH: System.out.print("v"); break;
						case WEST:  System.out.print("<"); break;
						case EAST:  System.out.print(">"); break;
						case IDLE:  System.out.print("."); break;
					}
				}
			}
			System.out.println();
		}
	}
}
