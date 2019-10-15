FROM ubuntu:16.04
FROM nvidia/cuda:8.0-runtime-ubuntu16.04

# Install some basic utilities
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
 && rm -rf /var/lib/apt/lists/*

RUN apt-get update
RUN apt-get install -y wget gcc curl xz-utils git xvfb libfontconfig wkhtmltopdf make
#-----------------------------------
# Install ffmpeg
RUN wget -P /tmp https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-64bit-static.tar.xz
RUN cd /tmp && \
    tar xf ffmpeg-release-64bit-static.tar.xz && \
    ls && \
    cd ffmpeg-4.1-64bit-static/ && \
    cp ffmpeg /usr/bin/ffmpeg && \
    cp ffprobe /usr/bin/ffprobe && \
    chmod +x /usr/bin/ffmpeg && \
    chmod +x /usr/bin/ffprobe

# convention for module location
RUN mkdir /code
WORKDIR /code

# Create a non-root user and switch to it
#RUN adduser --disabled-password --gecos '' --shell /bin/bash user && chown -R user:user /code
#RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
#USER user

# All users can use /home/user as their home directory
ENV HOME=/home/
RUN chmod 777 /home/

RUN curl -so ~/miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-4.5.1-Linux-x86_64.sh \
 && chmod +x ~/miniconda.sh \
 && ~/miniconda.sh -b -p ~/miniconda \
 && rm ~/miniconda.sh
ENV PATH=/home/miniconda/bin:$PATH
ENV CONDA_AUTO_UPDATE_CONDA=false

# Create a Python 3.6 environment
RUN /home/miniconda/bin/conda install conda-build \
 && /home/miniconda/bin/conda create -y --name py36 python=3.6.5 \
 && /home/miniconda/bin/conda clean -ya
ENV CONDA_DEFAULT_ENV=py36
ENV CONDA_PREFIX=/home/miniconda/envs/$CONDA_DEFAULT_ENV
ENV PATH=$CONDA_PREFIX/bin:$PATH

# CUDA 8.0-specific steps
RUN conda install -y -c pytorch \
    cuda80=1.0 \
    magma-cuda80=2.3.0 \
    "pytorch=0.4.1=py36_cuda8.0.61_cudnn7.1.2_1" \
    torchvision=0.2.1 \
 && conda clean -ya

# Install HDF5 Python bindings
RUN conda install -y h5py=2.8.0 \
 && conda clean -ya
RUN pip install h5py-cache==1.0

# install ffmpeg
#RUN conda install -c conda-forge ffmpeg
RUN pip install ffmpeg-python

# Install Torchnet, a high-level framework for PyTorch
RUN pip install torchnet==0.0.4

# Install sklearn for svm
RUN pip install numpy
RUN pip install scipy
RUN pip install scikit-learn==0.20.0

# convention for module code location
COPY app /code/app
RUN mkdir /code/tmp_frames
RUN mkdir /code/output
# convention for module model-related data location
# COPY model /code/model
# COPY app/resnet18-5c106cde.pth /home/user/.torch/models/resnet18-5c106cde.pth
# set entrypoint so that we could call the docker concisely in shell
CMD ["-h"]
# ENTRYPOINT ["/opt/conda/bin/python", "app/scene_classifier.py"]
# CMD ["python3"]
# ENTRYPOINT ["python3", "scene_app.py", "--v", "VIRAT_S_000000.mp4", "--m", "scene_19_clf_svm_linear.pkl"]
