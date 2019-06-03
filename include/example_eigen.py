import eigen_solver as es
import scipy.io as sio

# load mesh
model = 'sphere4.mat'
mat = sio.loadmat(model)
V, F = mat['V'], mat['F']

# solve eigen problem
output = es.steklov_eigen_solver(V,F,num_iter=25,top_k=49)
w_S, v_S = output.w_S, output.v_S
# w_S and v_S are resulting eigenvalues and eigenvectors.

# save results
folder = './result-' + model
es.save_eigen_ouput(folder,output)

# plot results, does not work for command line usage.
if False:
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(range(49), output.w_S)

# if running correctly, the w_S should approximately
# looks like [0,1,1,1,2,2,2,2,2,3,3,3,3,3,3,3,4...]