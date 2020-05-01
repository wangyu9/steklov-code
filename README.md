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

If the code runs correctly, you should expect output eigenvalues like 
[0,1,1,1,2,2,2,2,2,3,3,3,3,3,3,3,4,4,4,4,4,4,4,4,4,5...]

(Not exactly since the input is an polygon approximting the sphere. In my case, I got: 
[0.01004436 0.98603902 0.98855039 0.99042844 1.99039739 1.99604268
 1.99926872 2.00135786 2.00480661 3.00011596 3.00434902 3.00624768
 3.00751291 3.00917211 3.01196417 3.0124851  4.00351974 4.00446321
 4.00626848 4.00839618 4.00995248 4.01127969 4.01239172 4.01256993
 4.01452445 5.00554734 5.00791356 5.00822096 5.00949137 5.0104079
 5.01165094 5.01183372 5.01255543 5.01390004 5.01497264 5.01618703
 6.00388227 6.0089286  6.00975284 6.01054685 6.01299514 6.01395933
 6.01752202 6.01898825 6.01980104 6.02056086 6.02172131 6.02300891
 6.02432555]


Email wangyu9@mit.edu for any questions with the code.
