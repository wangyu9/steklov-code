FROM ubuntu:16.04
#FROM scratch
# FROM bempp/dev
#FROM python:2.7-slim
#FROM ubuntu:16.04


RUN apt-get update && \
      apt-get -y install sudo
RUN sudo apt-get update

# RUN sudo apt-get --yes install software-properties-common python-software-properties
# RUN sudo add-apt-repository ppa:bemppsolutions/bempp

# bempp # somehow the python-bempp cannot be located
# RUN sudo apt-get install python-bempp


# follow the instruction on the page http://www.bempp.org/installation.html


# Install wget and build-essential
RUN apt-get update && apt-get install -y \
  build-essential \
  wget \
  sudo \
  apt-utils

############## BASIC DEPENDENCIES AND COMMON PKGs #####################
# Install vim & nano
# RUN sudo apt-get --assume-yes install vim
RUN sudo apt-get --assume-yes install nano

# More basic dependencies and python 2.7
RUN apt-get update && apt-get install -y --no-install-recommends \
        apt-utils \
        module-init-tools \
        openssh-server \
        curl \
        libfreetype6-dev \
        libpng12-dev \
        libzmq3-dev \
        pkg-config \
        # python2.7 \
        # python2.7-dev \
	python \
	python-dev \
	python-pip \
	python-tk \
	python-lxml \
	python-six \
        rsync \
        software-properties-common \
        unzip \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Pip
# RUN curl -fsSL -O https://bootstrap.pypa.io/get-pip.py && \
#    python2.7 get-pip.py && \
#    rm get-pip.py

RUN sudo apt-get update && \
    sudo apt-get install -y autoconf libtool pkg-config

RUN sudo pip install --upgrade pip
RUN sudo pip install -U setuptools

# Jupyner and common python packages
RUN sudo pip --no-cache-dir install \
        ipykernel \
        jupyter \
        matplotlib \
        numpy \
        scipy \
        sklearn \
        Pillow
# sklearn somehow fails.

################ Dependencies for BEMPP #################
# http://www.bempp.org/installation.html

RUN sudo apt-get install zlib1g-dev
RUN sudo apt-get --assume-yes install libboost-all-dev

RUN sudo apt-get --assume-yes install git-all
RUN sudo apt-get --assume-yes install cmake

RUN sudo apt-get --assume-yes install doxygen
RUN sudo apt-get --assume-yes install libtbb-dev
RUN sudo apt-get install patchelf
RUN sudo apt-get --assume-yes install libdune-pdelab-dev
RUN sudo apt-get --assume-yes install libdune-pdelab-doc

RUN sudo pip install Cython

RUN sudo apt-get --assume-yes install gcc-4.7 g++-4.7

RUN sudo mkdir libbempp



# RUN git clone https://github.com/bempp/bempp.git
# RUN bash -c "cd bempp && git checkout ca21a9770e64b769645305e867d31fc5fa8a2e35"
RUN sudo git clone https://bitbucket.org/bemppsolutions/bempp.git
RUN bash -c "cd bempp && git checkout 310648ee26b7e96badc39b343925240e4d23d565"
# 3.1.4

# RUN bash -c "export CC=/usr/bin/gcc-4.7 && export CXX=/usr/bin/g++-4.7 && export CC=gcc-4.7 && export CXX=g++-4.7"
# RUN sudo git clone https://bitbucket.org/bemppsolutions/bempp.git
# RUN sudo cd bempp
# RUN bash -c "export CC=/usr/bin/gcc-4.7 && export CXX=/usr/bin/g++-4.7 && export CC=gcc-4.7 && export CXX=g++-4.7 &&  cd bempp && mkdir build && cd build && cmake -DCMAKE_INSTALL_PREFIX=/libbempp .. && make -j4 && make install"
RUN bash -c "cd bempp && mkdir build && cd build && cmake -DCMAKE_INSTALL_PREFIX=/libbempp .. && make -j4 && make install"


RUN sudo pip install mpi4py

# RUN sudo mkdir build
# RUN sudo cd build
# RUN bash -c "cd build"
# RUN sudo cmake -DCMAKE_INSTALL_PREFIX=/libbempp .. 

# export will not work
# RUN bash -c  "export PYTHONPATH=/libbempp/lib/python2.7/site-packages"
# Rsh -c  "export PYTHONPATH=/libbempp/lib/python2.7/site-packages"
# CMD bash -c "echo /libbempp/lib/python2.7/site-packages > bempp.pth"
# CMD sh -c  "ln -s /libbempp/lib/python2.7/site-packages/bempp /usr/local/lib/python2.7/site-packages/bempp" && /bin/bash
# https://stackoverflow.com/questions/44755187/setting-pythonpath-in-ubuntu-16-04-for-a-docker-image-to-run-properly
ENV PYTHONPATH /libbempp/lib/python2.7/site-packages

COPY ./ /steklov-core
