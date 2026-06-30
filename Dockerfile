FROM tensorflow/tensorflow:2.17.0-gpu

ENV DEBIAN_FRONTEND=noninteractive

# Dependencies
RUN apt-get update && apt-get install -y \
    software-properties-common \
    wget \
    curl \
    git \
    build-essential \
    cmake \
    libboost-all-dev \
    python3-pip \
    libgl1 \
    libxrender1 \
    libxkbcommon-x11-0 \
    libxi6 \
    libxxf86vm1 \
    libxfixes3 \
    libxcursor1 \
    libxrandr2 \
    libxinerama1 \
    libegl1 \
    libsm6 \
    libgmp-dev \
    libmpfr-dev \
    libglib2.0-0 \
    libgtk-3-0 \
    libgdk-pixbuf2.0-0 \
    libpango-1.0-0 \
    libcairo-gobject2 \
    libgtk-3-0 \
    libgdk-pixbuf2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default (already present in base image)
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

RUN python3.11 -m pip install --upgrade pip

# IMPORTANT to keep those versions, compatibility of most of the libraries depend on that
RUN python3.11 -m pip install tensorflow==2.17.0 keras==3.4.1

# Install all Python dependencies in one go
COPY requirements.txt /workspace/requirements.txt
COPY DSMNet/requirement.txt /workspace/DSMNet/requirement.txt
RUN python3.11 -m pip install -r /workspace/requirements.txt
RUN python3.11 -m pip install -r /workspace/DSMNet/requirement.txt

# CGAL 6.0.1
WORKDIR /opt
RUN git clone https://github.com/CGAL/cgal.git && \
    cd cgal && \
    git checkout v6.0.1 && \
    cmake -S . -B build && \
    cmake --build build --target install

# Blender 4.4.0
WORKDIR /opt
ENV BLENDER_VERSION=4.4.0
RUN wget https://download.blender.org/release/Blender4.4/blender-${BLENDER_VERSION}-linux-x64.tar.xz && \
    tar -xf blender-${BLENDER_VERSION}-linux-x64.tar.xz && \
    ln -s /opt/blender-${BLENDER_VERSION}-linux-x64/blender /usr/local/bin/blender && \
    rm blender-${BLENDER_VERSION}-linux-x64.tar.xz

# Install Blender Python packages
RUN /opt/blender-4.4.0-linux-x64/4.4/python/bin/python3.11 -m pip install plyfile shapely geopandas trimesh scipy laspy numpy networkx rtree

WORKDIR /workspace

# Create output directories
RUN mkdir -p /workspace/data/input/ortho \
    /workspace/data/input/footprints \
    /workspace/data/output/dsm \
    /workspace/data/output/rooftype \
    /workspace/data/output/3d_models

# Set environment variables
ENV PYTHONPATH=/workspace

# Entrypoint script to install dependencies and run the pipeline
RUN echo '#!/bin/bash\n\
if [ -f /workspace/requirements.txt ]; then\n\
    pip install -r /workspace/requirements.txt\n\
fi\n\
\n\
if [ -f /workspace/pipeline.sh ]; then\n\
    chmod +x /workspace/pipeline.sh\n\
    exec /workspace/pipeline.sh "$@"\n\
else\n\
    exec "$@"\n\
fi' > /entrypoint.sh && chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
