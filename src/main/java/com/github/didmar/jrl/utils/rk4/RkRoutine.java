package com.github.didmar.jrl.utils.rk4;

public interface RkRoutine<T> {
	public T rkRoutine(T X, double t, double dt);
}
