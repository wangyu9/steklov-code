# Steklov Spectral Geometry for Extrinsic Shape Analysis 

This repo contains an implementation of the paper 


*	**Steklov Spectral Geometry for Extrinsic Shape Analysis**.

	Yu Wang, Mirela Ben-Chen, Iosif Polterovich and Justin Solomon.

	_ACM Transactions on Graphics 38(1)._ _(Presented at) ACM SIGGRAPH 2019_.
 
	[OpenAccessPaper](https://dl.acm.org/citation.cfm?id=3152156).\\


# Update 2024.8.2
We have upgraded the code to be compatible with python 3.10 and the newest bempp-cl.  You can run our code in the official bempp-cl docker image
```shell
docker pull bempp/cl-notebook:latest
docker run -v /path/to/folder:/root/shared -p 8888:8888 bempp/cl-notebook
```
or install bempp-cl by yourself, by following the official instruction: https://bempp.com/installation.html

Note the backend of bempp-cl is based on the fast multipole method (FMM), which is different from the older version of bempp we used in the paper that has the hierarchical matrix (HM) backend. 

# Old version setup instruction
Originally, the code is written in python 2.7. To use this version: 
```shell
git checkout d86944f
```

We provide a docker container for ease of setting up dependencies. (Docker is like a light-weight virtual machine so you don't need to install anything except docker.) Alternatively you can install all depenciencies manually following the Dockerfile. 

To build the docker image, run:
```shell
docker build -t steklov-py2 .
```

And start the container as:
```shell
docker run -it steklov-py2
```

The docker file was simply following the official compiling guide of (an earlier version of) bempp. The guide was gone as they upgrade to a new version. I also packed and uploaded my compiled docker image here: https://www.dropbox.com/s/ywk88x1nmc4423p/steklov-docker.tar?dl=0
You can load the docker image directly using, e.g.:
```shell
docker load -i <path to docker image tar file>
```
You can also look at the versions of dependencies etc in the docker image. 

# Usage

As a starting example, in the docker container, the following code 
```shell
cd steklov-core/include
python example_eigen.py
```
will solve the Steklov eigenvalue problem on a test mesh (unit sphere).  

If the code runs correctly, you should expect output eigenvalues that look like 
```
[0,1,1,1,2,2,2,2,2,3,3,3,3,3,3,3,4,4,4,4,4,4,4,4,4,5...]
```

(Not exactly the numerics since the input is an polygon mesh approximating the sphere. In my case (py2 version), I got: 
```
[0.01004436 0.98603902 0.98855039 0.99042844 1.99039739 1.99604268
 1.99926872 2.00135786 2.00480661 3.00011596 3.00434902 3.00624768
 3.00751291 3.00917211 3.01196417 3.0124851  4.00351974 4.00446321
 4.00626848 4.00839618 4.00995248 4.01127969 4.01239172 4.01256993
 4.01452445 5.00554734 5.00791356 5.00822096 5.00949137 5.0104079
 5.01165094 5.01183372 5.01255543 5.01390004 5.01497264 5.01618703
 6.00388227 6.0089286  6.00975284 6.01054685 6.01299514 6.01395933
 6.01752202 6.01898825 6.01980104 6.02056086 6.02172131 6.02300891
 6.02432555]
```

Email wangyu9@mit.edu for any question with the code.
