package com.github.didmar.jrl.environment.cartpole;

import java.awt.Color;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.geom.Rectangle2D;

import com.github.didmar.jrl.utils.plot.environment.EnvironmentDisplay;

/**
 * Listen to a {@link CartPole} environment and display its state.
 * @author Didier Marin
 */
public final class CartPoleDisplay extends EnvironmentDisplay {

	private static final long serialVersionUID = 7977449256500391149L;

	public static final double DEFAULT_ZOOM = 100.;

	public static final double poleWidth = CartPole.l / 10.;
	public static final double cartHeight = poleWidth;
	public static final double cartWidth = cartHeight * 10;

	private int scaledCartCenterX = 0;
	private int scaledCartWidth = 0;
	private int scaledCartHeight = 0;
	private int scaledCartMaxPos = 0;
	private int poleX = 0;
	private int poleY = 0;
	private final Rectangle2D.Double cart = new Rectangle2D.Double();
	private double currentReward = 0;

	public CartPoleDisplay() {
		super(DEFAULT_ZOOM);
	}

	public CartPoleDisplay(double zoom) {
		super(zoom);
	}

	/* (non-Javadoc)
	 * @see jrl.environment.EnvironmentListener#newEpisode(double[], int)
	 */
	@Override
	public final void newEpisode(double[] x0, int maxT) {
		update(x0,0.);
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
		scaledCartWidth = (int) (cartWidth * zoom);
		scaledCartHeight = (int) (cartHeight * zoom);
		scaledCartMaxPos = (int) (CartPole.maxPos * zoom);
		scaledCartCenterX  = (int)(x[CartPole.CART_POSITION] * zoom);
		cart.setRect(scaledCartCenterX-scaledCartWidth/2,
				0, scaledCartWidth, scaledCartHeight);
		double theta = x[CartPole.POLE_POSITION] + Math.PI/2.;
		poleX = (int) (Math.cos(theta) * CartPole.l * zoom);
		poleY = (int) (Math.sin(theta) * CartPole.l * zoom);
		currentReward  = r;
		repaint();
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
		Rectangle2D.Double centeredCart
			= new Rectangle2D.Double(centerX+cart.x, centerY+cart.y,
				cart.width, cart.height);
		// Draw the cart
		g2d.draw(centeredCart);
		// Draw the pole
		if(currentReward > 0.) {
			g2d.setColor(Color.GREEN);
		} else if (currentReward < 0) {
			g2d.setColor(Color.RED);
		}
		g2d.drawLine((int)centerX+scaledCartCenterX, (int)centerY,
				(int)centerX+scaledCartCenterX+poleX, (int)centerY-poleY);
		g2d.setColor(Color.BLACK);
		// Draw the ground
		g2d.drawLine((int)(centerX-scaledCartMaxPos-scaledCartWidth/2),
				(int)(centerY+cart.height),
				(int)(centerX+scaledCartMaxPos+scaledCartWidth/2),
				(int)(centerY+cart.height));
		// Draw the "walls" (cart position bounds)
		g2d.drawLine((int)(centerX-scaledCartMaxPos-scaledCartWidth/2),
				(int)(centerY-cart.height),
				(int)(centerX-scaledCartMaxPos-scaledCartWidth/2),
				(int)(centerY+cart.height));
		g2d.drawLine((int)(centerX+scaledCartMaxPos+scaledCartWidth/2),
				(int)(centerY-cart.height),
				(int)(centerX+scaledCartMaxPos+scaledCartWidth/2),
				(int)(centerY+cart.height));
	}
}
