package com.github.didmar.jrl.utils.rk4;

import com.github.didmar.jrl.utils.array.ArrUtils;

public class RungeKutta4<T> {
	@SuppressWarnings("unchecked")
	public final T rk4(T y, double t, double dt, int dir, RkRoutine<T> rkRoutine) {
		
		double th, hh, h6, h;
	    T dydx, dym, dyt, yt;
	    
	    h = dt * dir;
	    hh = h * 0.5;
	    h6 = h / 6.0;
	    th = t + hh;

	    dydx = rkRoutine.rkRoutine(y, t, dt);
	    
	    // yt = y + hh * dydx;
	    if(y instanceof double[]) {
	    	yt = (T) new double[((double[])y).length];
	    	double[] ytCasted = ((double[])yt);
	    	for (int i = 0; i < ytCasted.length; i++) {
				ytCasted[i] = ((double[])y)[i] + hh * ((double[])dydx)[i];
			}
	    } else {
	    	yt = (T) ArrUtils.emptyLike((double[][])y);
	    	double[][] ytCasted = ((double[][])yt);
	    	for (int i = 0; i < ytCasted.length; i++) {
	    		for (int j = 0; j < ytCasted[0].length; j++) {
	    			ytCasted[i][j] = ((double[][])y)[i][j] + hh * ((double[][])dydx)[i][j];
	    		}
			}
	    }
	    dyt = rkRoutine.rkRoutine(yt, th, dt);
	    // yt = y + hh * dyt;
	    if(y instanceof double[]) {
	    	double[] ytCasted = ((double[])yt);
	    	for (int i = 0; i < ytCasted.length; i++) {
				ytCasted[i] = ((double[])y)[i] + hh * ((double[])dyt)[i];
			}
	    } else {
	    	double[][] ytCasted = ((double[][])yt);
	    	for (int i = 0; i < ytCasted.length; i++) {
	    		for (int j = 0; j < ytCasted[0].length; j++) {
	    			ytCasted[i][j] = ((double[][])y)[i][j] + hh * ((double[][])dyt)[i][j];
	    		}
			}
	    }
	    dym = rkRoutine.rkRoutine(yt, th, dt);
	    // yt = y + h * dym;
	    // dym += dyt;
	    if(y instanceof double[]) {
	    	double[] ytCasted = ((double[])yt);
	    	for (int i = 0; i < ytCasted.length; i++) {
				ytCasted[i] = ((double[])y)[i] + hh * ((double[])dym)[i];
				((double[])dym)[i] += ((double[])dyt)[i];
			}
	    } else {
	    	double[][] ytCasted = ((double[][])yt);
	    	for (int i = 0; i < ytCasted.length; i++) {
	    		for (int j = 0; j < ytCasted[0].length; j++) {
	    			ytCasted[i][j] = ((double[][])y)[i][j] + hh * ((double[][])dym)[i][j];
	    			((double[][])dym)[i][j] += ((double[][])dyt)[i][j];
	    		}
			}
	    }
	    dyt = rkRoutine.rkRoutine(yt, t + h, dt);
	    // res = y + h6 * (dydx + dyt + 2 * dym);
	    if(y instanceof double[]) {
	    	double[] res = new double[((double[])y).length];
	    	for (int i = 0; i < res.length; i++) {
				res[i] = ((double[])y)[i] + h6 * (((double[])dydx)[i] + ((double[])dyt)[i] + 2. * ((double[])dym)[i]);
			}
	    	return (T)res;
	    } else {
	    	double[][] res = new double[((double[][])y).length][((double[][])y)[0].length];
	    	for (int i = 0; i < res.length; i++) {
	    		for (int j = 0; j < res[0].length; j++) {
	    			res[i][j] = ((double[][])y)[i][j] + h6
	    					* (((double[][])dydx)[i][j] + ((double[][])dyt)[i][j]
	    							+ 2. * ((double[][])dym)[i][j]);
	    		}
			}
	    	return (T)res;
	    }
	}
}