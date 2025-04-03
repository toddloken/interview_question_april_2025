# Use Miniconda base image
FROM continuumio/miniconda3

# Set working directory
WORKDIR /app

# Copy your project files into the container
COPY . /app

# Create the environment
RUN sed -i '/win_inet_pton/d' conda_environment.yml && \
    sed -i '/intel-openmp/d' conda_environment.yml && \
    sed -i '/libwinpthread/d' conda_environment.yml && \
    sed -i '/ucrt/d' conda_environment.yml && \
    sed -i '/vc/d' conda_environment.yml

# Activate the environment and ensure it's used by default
SHELL ["conda", "run", "-n", "housing", "/bin/bash", "-c"]

# Optional: install conda-pack to lock environment
# RUN conda install -n housing conda-pack

# Default command to run your a
CMD ["conda", "run", "-n", "housing", "python", "app.py"]
