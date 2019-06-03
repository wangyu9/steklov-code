# Steklov Spectral Geometry for Extrinsic Shape Analysis 

This repo contains an implementation of the paper ``Steklov Spectral Geometry for Extrinsic Shape Analysis 
''

by Yu Wang, Mirela Ben-Chen, Iosif Polterovich and Justin Solomon 

ACM Transactions on Graphics 38(1), arXiv:1707.07070

https://arxiv.org/abs/1707.07070.

We provide a docker container for ease of setting up dependencies. Alternatively you can install all depenciencies manually following the Dockerfile. 

To build the docker image, run:
```shell
sudo docker build -t steklov-py2 .
```

And start the container as:
```shell
sudo docker run -it steklov-py2
```
As a starting example, in the docker container, the following code 
```shell
cd steklov-core/include
python example_eigen.py
```
will solve the Steklov eigenvalue problem on a test mesh (unit sphere).  

Email wangyu9@mit.edu for any questions with the code.
