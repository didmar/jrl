package com.github.didmar.jrl.environment.acrobot;

import java.awt.Color;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.geom.Rectangle2D;

import com.github.didmar.jrl.utils.plot.environment.EnvironmentDisplay;

public final class AcrobotDisplay extends EnvironmentDisplay {

private static final long serialVersionUID = 7977449256500391149L;

	public static final double DEFAULT_ZOOM = 100.;

	private int scaledElbowX = 0;
	private int scaledElbowY = 0;
	private int scaledTipX = 0;
	private int scaledTipY = 0;
	private int scaledGoalHeight = 0;
	private double currentReward = 0;

	private final Acrobot acrobot;

	public AcrobotDisplay(Acrobot acrobot) {
		super(DEFAULT_ZOOM);
		this.acrobot = acrobot;
	}

	public AcrobotDisplay(double zoom, Acrobot acrobot) {
		super(zoom);
		this.acrobot = acrobot;
	}

	/* (non-Javadoc)
	 * @see jrl.environment.EnvironmentListener#newEpisode(double[], int)
	 */
	@Override
	public final void newEpisode(double[] x0, int maxT) {
		update(x0,-1.);
	}

	/* (non-Javadoc)
	 * @see jrl.environment.EnvironmentListener#receiveSample(double[], double[], double[], double)
	 */
	@Override
	public final void receiveSample(double[] x, double[] u, double[] xn, double r, boolean isTerminal) {
		update(x,r);
	}

	/* (non-Javadoc)
	 * @see jrl.environment.EnvironmentListener#endEpisode()
	 */
	@Override
	public final void endEpisode() {
		// Nothing to do
	}

	private final void update(double[] x, double r) {
		scaledElbowX = (int)(  Math.sin(x[0])*Acrobot.l1 * zoom);
		scaledElbowY = (int)( -Math.cos(x[0])*Acrobot.l1 * zoom);
		scaledTipX   = (int)(( Math.sin(x[0])*Acrobot.l1 + Math.sin(x[0]+x[1])*Acrobot.l2) * zoom);
		scaledTipY   = (int)((-Math.cos(x[0])*Acrobot.l1 - Math.cos(x[0]+x[1])*Acrobot.l2) * zoom);
		scaledGoalHeight = (int)(acrobot.goalHeight * zoom);
		currentReward  = r;
		repaint();
		try {
			Thread.sleep(100);
		} catch (InterruptedException e) {
			e.printStackTrace();
		}
	}

	@Override
	public final void paintComponent(Graphics g) {
		// super.paintComponent clears off screen pixmap,
		// since we're using double buffering by default.
		super.paintComponent(g);
		Graphics2D g2d = (Graphics2D) g;
		Rectangle2D panelBounds = g2d.getClipBounds();
		double centerX = panelBounds.getCenterX();
		double centerY = panelBounds.getCenterY();
		// Draw the segments
		if(currentReward < 0.) {
			g2d.setColor(Color.RED);
		} else {
			g2d.setColor(Color.GREEN);
		}
		g2d.drawLine((int)centerX, (int)centerY,
				(int)centerX+scaledElbowX, (int)centerY-scaledElbowY);
		g2d.drawLine((int)centerX+scaledElbowX, (int)centerY-scaledElbowY,
				(int)centerX+scaledTipX, (int)centerY-scaledTipY);
		// Draw the goal bar
		g2d.setColor(Color.BLACK);
		g2d.drawLine((int)(centerX-(Acrobot.l1*zoom)/2.),
				(int)(centerY-scaledGoalHeight),
				(int)(centerX+(Acrobot.l1*zoom)/2.),
				(int)(centerY-scaledGoalHeight));
	}

}
