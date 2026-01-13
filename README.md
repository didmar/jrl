# jRL #

jRL is a Java library for Reinforcement Learning,
developped by Didier Marin (mail@didiermarin.com)
during his PhD.

It is focused on continuous multi-dimensional states and actions.

## Dependencies ##

- For Least-Squares : JAMA 1.0.2
- For plotting : gnuplot
- For Unit tests : JUnit 4
- For @NonNull and @Nullable annotations : org.eclipse.jdt.annotation.jar

## Running with Docker ##

Build the Docker image:

```bash
docker build -t jrl .
```

The image includes Java 8, Maven, gnuplot, and Python 3 with matplotlib/numpy.

### Run Tests

```bash
# Run all utility tests
docker run --rm jrl mvn test -Dtest=TestArrUtils,TestUtils,TestRandUtils

# Run a specific test
docker run --rm jrl mvn test -Dtest=TestArrUtils
```

### Run Examples

```bash
# Least Squares example (non-interactive)
docker run --rm jrl mvn exec:java -Dexec.mainClass="com.github.didmar.jrl.examples.ExLeastSquares"
```

For GUI-based examples (ExFeatures, ExHumanAgent), you need X11 forwarding:

```bash
# If X11 is your display server:
docker run -it --rm \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  jrl mvn exec:java -Dexec.mainClass="com.github.didmar.jrl.examples.ExHumanAgent"

# If Wayland is your display server:
xhost +local:docker  # Allow local connections (run on host)
docker run -it --rm \
  -e DISPLAY=$DISPLAY \
  -e WAYLAND_DISPLAY=$WAYLAND_DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v $XDG_RUNTIME_DIR/$WAYLAND_DISPLAY:$XDG_RUNTIME_DIR/$WAYLAND_DISPLAY \
  jrl mvn exec:java -Dexec.mainClass="com.github.didmar.jrl.examples.ExHumanAgent"
```

### Plot Performance Data

The `plot_perf.py` script visualizes benchmark results:

```bash
docker run --rm -v "$(pwd):/app" -w /app jrl \
  python3 plot_perf.py "Algorithm 1" perf_file1 "Algorithm 2" perf_file2
```

In headless mode, the plot is saved to `performance_plot.png`.

### Interactive Shell

```bash
docker run -it --rm jrl bash
```
