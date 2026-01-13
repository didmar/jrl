FROM eclipse-temurin:8-jdk

# Install system dependencies
RUN apt-get update && apt-get install -y \
    maven \
    gnuplot \
    python3 \
    python3-pip \
    python3-matplotlib \
    python3-numpy \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy pom.xml first to cache dependencies
COPY pom.xml .

# Download all dependencies and plugins
RUN mvn dependency:go-offline dependency:resolve-plugins

# Copy the rest of the project
COPY . .

# Build the project and download exec plugin
RUN mvn package -DskipTests && \
    mvn exec:help -q

# Default command
CMD ["bash"]
