package com.github.didmar.jrl.environment.dynsys;

public class DynSysException extends Exception {

	private static final long serialVersionUID = 2867035518855456751L;

	final String msg;

	public DynSysException(String msg) {
		this.msg = msg;
	}

	@Override
	public String toString() {
		return msg;
	}
}
