package com.github.didmar.jrl.utils;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintStream;

import org.eclipse.jdt.annotation.Nullable;

/**
 * This class is used to communicate with an external program using standard
 * input and output.
 * @author Didier Marin
 */
public class Console {
	
	public final String executable;

	// communication channel: console.output -> process.input
	private final PrintStream consoleIn;
	private final BufferedReader consoleOut;

	@Nullable private String exitCommand;

	/**
	 * Default constructor executes a OS-specific command to start an external
	 * program and establishes the communication.
	 * @throws IOException if the system fails to execute the external program
	 */
	public Console(String executable) throws IOException {
		this.executable = executable;
		// start the process and connect channels
		Process p = Runtime.getRuntime().exec(executable);
		consoleIn = new PrintStream(p.getOutputStream());
		consoleOut = new BufferedReader(new InputStreamReader(p.getInputStream()));
		exitCommand = null;
	}

	/**
	 * Sends the given <code>command</code> to the external program.
	 * @param command    the command to execute on the external process
	 */
	public final void execute(String command) {
		consoleIn.println(command);
		consoleIn.flush();
	}
	
	public final String readOutput() throws IOException {
		String s = "";
		String line;
		while ((line = consoleOut.readLine()) != null) {
			s += line+"\n";
		}
		return s;
	}
	

	/**
	 * Exit the external program and close the in/out streams.
	 */
	public final void close() {
		if (exitCommand != null) {
			this.execute(exitCommand);
			consoleIn.close();
		}
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see java.lang.Object#finalize()
	 */
	@Override
	protected final void finalize() throws Throwable {
		this.close();
		super.finalize();
	}
	
	public final String getExecutable() {
		return executable;
	}
	
	@Nullable
	public final String getExitCommand() {
		return exitCommand;
	}

	public final void setExitCommand(String exitCommand) {
		this.exitCommand = exitCommand;
	}
}
