import bempp_operator as op
# import igl_viewer as gx

import scipy
import scipy.io as sio
from scipy import linalg as SLA
from scipy.sparse import linalg as SSLA
import scipy.sparse as sp
import numpy as np

import scipy.spatial.distance as dist
import matplotlib.pyplot as plt

import bempp.api

from numpy import linalg as LA

import optimization_util as opt

#############################
# http://stackoverflow.com/questions/5136611/capture-stdout-from-a-script-in-python
'''import contextlib
@contextlib.contextmanager
def Capturing(doit):
    if not doit:
        return
    import sys
    from cStringIO import StringIO
    oldout,olderr = sys.stdout, sys.stderr
    try:
        out=[StringIO(), StringIO()]
        sys.stdout,sys.stderr = out
        yield out
    finally:
        sys.stdout,sys.stderr = oldout, olderr
        out[0] = out[0].getvalue()
        out[1] = out[1].getvalue()
'''
#http://stackoverflow.com/questions/16571150/how-to-capture-stdout-output-from-a-python-function-call
from cStringIO import StringIO
import sys
class Capturing(list):
    def __init__(self, doit):
        self._doit = doit

    def __enter__(self):
        self._stdout = sys.stdout
        if self._doit:
            sys.stdout = self._stringio = StringIO()
        return self

    def __exit__(self, *args):
        if self._doit:
            self.extend(self._stringio.getvalue().splitlines())
            del self._stringio  # free up some memory
        sys.stdout = self._stdout


#########################################
# Functions
#########################################

reload(opt)

'''
# removed the show_mesh and the mayavi dependency.
def show_mesh(V,F,u):
    from mayavi import mlab
    mlab.figure(size=(1200,1200))
    mlab.triangular_mesh(V[:, 0], V[:, 1], V[:, 2], F, scalars=u), mlab.colorbar(), mlab.show()
    #mlab.view(azimuth=0, elevation=90, distance=2)
'''

def mat(M):
    return bempp.api.as_matrix(M)

def d(M):
    n = M.shape[0]
    m = M.shape[1]
    dM = np.zeros([n,m])
    for j in range(m):
        v = np.zeros(m)
        v[j] = 1
        dM[:,j] = M.matvec(v)
    return dM

def check_symmetry(M,epsilon=1e-6):
    residual_norm = LA.norm(M-M.transpose())
    assert(residual_norm<epsilon)
    return residual_norm

def double_area(V,F):

    a = np.linalg.norm( V[F[:, 1], :] - V[F[:, 2], :], axis=1)
    b = np.linalg.norm( V[F[:, 2], :] - V[F[:, 0], :], axis=1)
    c = np.linalg.norm( V[F[:, 0], :] - V[F[:, 1], :], axis=1)
    s = (a+b+c) / 2

    dblA = np.sqrt(s*(s-a)*(s-b)*(s-c)) * 2 # since it is double area.

    return dblA

def galerkin_weights(V,F):
    assert (F.shape[1] == 3)
    dblA = double_area(V, F) # no longer use the igl one.
    n = V.shape[0]
    weights = np.zeros(n)
    for i in range(F.shape[0]):
        for j in range(3):
            weights[F[i, j]] = weights[F[i, j]] + dblA[i]
    weights = weights / 3
    return weights

def single_layer_preconditioner(L,H,M,invM,weq):
    n = L.shape[0]

    lambda_w = 1/np.sum( weq * M.matvec(np.ones(n)))

    wHt = invM * H * invM + lambda_w / 4 * opt.array2clo(weq) * opt.array2rlo(weq)

    return 4*wHt

def check_matvec(M):
    op.tic()
    M.matvec(np.ones(M.shape[1]))
    op.toc()
    return

def check_rmatvec(M):
    op.tic()
    M.rmatvec(np.ones(M.shape[1]))
    op.toc()
    return

def check_matmat(M):
    op.tic()
    M.matmat(np.ones([M.shape[1],10]))
    op.toc()
    return

def check(b):
    import warnings
    if not b:
        warnings.warn('Asseratin Failed', Warning)

class C(object):
    def __init__(self, keys='', values=[]):
        for (key, value) in zip(keys, values):
            self.__dict__[key] = value


def steklov_eigen_solver(V,F,*positional_parameters, **keyword_parameters):

    try:
        op.tic()
        op.toc(silent=True)
    except Exception, e:
        op.toc(silent=True)

    if('input' in keyword_parameters):
        input = keyword_parameters['input']
    else:
        input = C('', [])

    for kp in keyword_parameters:
        if kp != 'input':
            setattr(input, kp, keyword_parameters[kp])

    def input_parser(str, default_value):
        return getattr(input, str, default_value)

    capture = input_parser('capture', False)

    with Capturing(capture) as stdout:

        # parsing option

        verify_dense = input_parser('verify_dense',False) # warning: full matrix, very slow
        debug_mode = input_parser('debug_mode',False) # warning: very slow
        check_cond_number = input_parser('check_cond_number',False) # warning: very slow
        run_slp = input_parser('run_slp', False) # not necessary

        top_k = input_parser('top_k',49)
        hmat_eps = input_parser('hmat_eps', 1e-2) # set to 1e-3 for better accuracy.
        num_iter = input_parser('num_iter', 20)

        pre = input_parser('pre', []) # pre contains initialization of eigenvectors, can be obtained by multigrid methods.

        dense_solve = input_parser('dense_solve', False)

        # do not change this unless you know what you are doing.
        weighted = input_parser('weighted', True)
        lumped_mass = input_parser('lumped_mass', False)
        natural_density_slp = False

        #########################################
        # Initial Operators
        #########################################
        print 'top_k: %d\n'%top_k
        print 'num_iter: %d\n'%num_iter
        print 'hmat_eps: %f\n'%hmat_eps

        output = C()
        output.timings = C()
        output.V = V
        output.F = F
        output.input = input

        outpre = C()

        n = V.shape[0]

        print "Assembling Operators...\n"
        op.tic()

        LL, KK, TT, HH, MM = op.symmetrized_boundary_operators(V, F, eps=hmat_eps) #, 'dense')

        # W = gx.galerkin_weights(V, F)
        W = galerkin_weights(V, F)

        if lumped_mass:
            # using a lumped mass matrix to speed up.
            print('Lumped mass option is currently unavailable.')
            assert(False)
            # M_sp = gx.mass_matrix(V, F)
            # M_sp = scipy.sparse.csc_matrix((M_sp.data, (M_sp.row, M_sp.col)), shape=M_sp.shape)
        else:
            M_sp = mat(MM)

        L, K, T, H = opt.bem2lo(LL), opt.bem2lo(KK), opt.bem2lo(TT), opt.bem2lo(HH)
        M = opt.s2lo(M_sp,M_sp)

        # by default the rmatvec is undefined.
        L.rmatvec = L.matvec
        H.rmatvec = H.matvec
        K.rmatvec = T.matvec
        T.rmatvec = K.matvec
        M.rmatvec = M.matvec

        if lumped_mass:
            invM = opt.diag2lo(1/M_sp.data)
        else:
            invM = opt.inv_sparse(M_sp,M_sp)

        print "Finished Assembling Operators."
        output.timings.assemble = op.toc()

        #########################################
        # Preconditioned SLP system
        #########################################

        ########### SLP Preconditioner ###########

        print "Computing SLP Precondition...\n"
        op.tic()

        if natural_density_slp:
            if dense_solve:
                weq = SLA.solve(d(L), M.matvec(np.ones(n)))
            else:
                if not pre:
                    weq, info = SSLA.cg(L,b=M.matvec(np.ones(n)),tol=1e-2)
                else:
                    weq, info = SSLA.cg(L,b=M.matvec(np.ones(n)),tol=1e-2, x0=outpre.weq)
        else:
            w_ones = np.ones(n)
            weq = w_ones/np.sqrt(np.inner(w_ones,M.matvec(w_ones)))

        output.weq = weq
        outpre.weq = weq

        invPL = single_layer_preconditioner(L,H,M,invM,weq)

        L.matmat = opt.naive_matmat(L.matvec)
        invPL.matmat = opt.naive_matmat(invPL.matvec)

        print "Finished Computing SLP Precondition."
        output.timings.slp_cg = op.toc()

        if debug_mode:
            print 'Check Linear Operators rmatvc:\n'
            check_rmatvec(L)
            check_rmatvec(H)
            check_rmatvec(K)
            check_rmatvec(T)
            check_rmatvec(M)
            check_rmatvec(invM)
            check_rmatvec(invPL)

            check_matvec(invPL)
            check_matmat(invPL)


        if verify_dense:
            dL = d(L)
            dM = d(M)
            dK = d(K)
            dT = d(T)
            dH = d(H)

            print 'error L: %d\n' % check_symmetry(dL)
            print 'error H: %d\n' % check_symmetry(dH)

        if verify_dense:

            dinvPL = d(invPL)
            print 'error invPL: %d\n' % check_symmetry(dinvPL)

            # check L is psd
            w_dL, v_dL = SLA.eigh(dL, b=dM)
            assert(w_dL[0]>0)

            # check the conditioning
            u_dinvPL_dL, s_dinvPL_dL, vt_dinvPL_dL = SLA.svd(np.dot(dinvPL, dL))
            print 'Cond(dinvPL*dL):%f\n' % (s_dinvPL_dL[0]/s_dinvPL_dL[-1])
            plt.plot(s_dinvPL_dL)

            output.w_dL = w_dL
            output.s_dinvPL_dL = s_dinvPL_dL

        def iter_cond_number_lobpcg(L, invPL):
            k = 20
            scipy.random.seed(0)
            X = scipy.random.rand(n, k)
            # compute the smallest eigenvalues:
            min_w_invPL_L, temp = SSLA.lobpcg(L * invPL * invPL * L, X, maxiter=20, tol=1e-3, largest=False, verbosityLevel=1)
            # compute the largest eigenvalues:
            max_w_invPL_L, temp = SSLA.lobpcg(opt.eyelo(n), X, B=L * invPL * invPL * L, maxiter=20, tol=1e-3, largest=False,
                                              verbosityLevel=1)
            max_w_invPL_L = 1 / max_w_invPL_L
            assert (max_w_invPL_L[0] >= min_w_invPL_L[::-1][0] and min_w_invPL_L[0] <= min_w_invPL_L[::-1][0])

            return np.sqrt(max_w_invPL_L[0] / min_w_invPL_L[0])



        if check_cond_number:
            print 'LOBPCG is not quite reliable Cond(dinvPL*dL):%f\n' % iter_cond_number_lobpcg(L, invPL)


        def iter_cond_number(L, invPL):
            max_w_invPL_L, temp = SSLA.eigsh(L * invPL * invPL * L, tol=1e-2, k=6)
            min_w_invPL_L, temp = SSLA.eigsh(L * invPL * invPL * L, tol=1e-2, k=6, which='SM')
            assert (max_w_invPL_L[::-1][0] >= max_w_invPL_L[0] and min_w_invPL_L[0] <= min_w_invPL_L[::-1][0])
            return np.sqrt(max_w_invPL_L[::-1][0] / min_w_invPL_L[0])

        if check_cond_number:
            cond_invPL_L = iter_cond_number(L, invPL)
            print 'Cond(invPL*L):%f\n' % cond_invPL_L
            output.cond_invPL_L = cond_invPL_L


        if 0: # this does not accept Minv somehow, try to update the Scipy
            w_invPL_L = SSLA.eigsh(L,k=6,Minv=invPL,which='SA',return_eigenvectors=False)


        ########### ARPACK eigensolver ###########
        # this comparison does not always work.
        if 0:
            w_L, v_L = SSLA.eigsh(L,k=6,which='SA',return_eigenvectors=True,maxiter=1000)

        ########### LOBPCG eigensolver ###########
	# this is the only reliable solver. 

        if run_slp:

            k = top_k
            # initial approximation to the k eigenvectors
            if not pre:
                scipy.random.seed(0)
                X = scipy.random.uniform(-1,1,size=(L.shape[0], k))
            else:
                X = pre.v_L
            largest = False
            w_L, v_L = SSLA.lobpcg(L, X, B=M, M=invPL, maxiter=10, tol=1e-8, largest=largest,  verbosityLevel=1)
            if largest:
                w_L = w_L[::-1]
                v_L = v_L[:,::-1]
            w_L = 1/w_L
            plt.figure()
            plt.plot(range(k), w_L)
            if verify_dense:
                plt.plot(range(k), w_dL[range(k)])

            # show_mesh(V,F,v_L[:,0])

            output.w_L = w_L
            output.v_L = v_L

            outpre.v_L = v_L
        #########################################
        # Preconditioned Calderon system
        #########################################

        ########### Steklov Preconditiner ###########

        invPS = invM * L * invM

        if debug_mode:
            check_rmatvec(invPS)

        # Set the right coefficients

        # this does not work unfortunately:
        # w_invPL_L, v_invPL_L = SSLA.eigsh(L,k=6,Minv=invPL)

        print "Computing gamma...\n"
        op.tic()

        k = 4
        if dense_solve:
            w_invPL_L, v_invPL_L = SLA.eigh(d(L * invPL * invPL * L))
            w_invPL_L = w_invPL_L[range(k)]
            v_invPL_L = v_invPL_L[:, range(k)]
        else:
            if 1:

                if not pre:
                    scipy.random.seed(0)
                    X = scipy.random.uniform(-1, 1, size=(n, k))
                else:
                    X = pre.v_invPL_L
                #w_invPL_L, v_invPL_L = SSLA.lobpcg(opt.transpose(invPL * L) * invPL * L, X, maxiter=5, tol=1e-5, largest=False,
                #                                   verbosityLevel=1)  # B=rhs,
                w_invPL_L, v_invPL_L = SSLA.lobpcg( L * invPL * invPL * L, X, maxiter=60, tol=1e-2, largest=False,
                                                   verbosityLevel=1)
            #else:
            #    w_invPL_L, temp = SSLA.eigsh(L * invPL * invPL * L, tol=1e-3, k=6, which='SM')
            #    s_invPL_L = np.sqrt(w_invPL_L)
            else:
                w_invPL_L, v_invPL_L = SSLA.eigsh(L*invPL*invPL*L, tol=1e-2, k=k, which='SM')
                assert (w_invPL_L[0] <= w_invPL_L[::-1][0])

        s_invPL_L = np.sqrt(w_invPL_L)
        outpre.v_invPL_L = v_invPL_L

        assert(s_invPL_L[0]<1)
        gamma = 0.9 * s_invPL_L[0]
        alpha = 1./gamma

        print "Finished Computing gamma."
        output.timings.gamma = op.toc()

        output.s_invPL_L = s_invPL_L
        output.gamma = gamma
        output.alpha = alpha

        ########### Assembling the System ###########

        A = opt.saddle_system_matrix(L,K+0.5*M,T+0.5*M,H,alpha*invPL)
        if weighted:
            rhs = opt.saddle_rhs_matrix(M)
        else:
            rhs = opt.saddle_rhs_matrix(opt.eyelo(M.shape[0]))
        invPA = opt.blkdiag([1./(alpha-1)*invPL,invPS])

        A.matmat = opt.naive_matmat(A.matvec)
        rhs.matmat = opt.naive_matmat(rhs.matvec)
        invPA.matmat = opt.naive_matmat(invPA.matvec)


        if debug_mode:
            check_matvec(A)
            check_matvec(rhs)
            check_matvec(invPA)

            check_rmatvec(A)
            check_rmatvec(rhs)
            check_rmatvec(invPA)

            check_matmat(A)
            check_matmat(rhs)
            check_matmat(invPA)

        if verify_dense:
            dA = d(A)
            dinvPA = d(invPA)
            drhs = d(rhs)

            print 'error invPA: %d\n' % check_symmetry(dinvPA)

            w_dinvPA, v_dinvPA = SLA.eigh(dinvPA)
            assert(w_dinvPA[0]>0)

            u_cond_dA, s_cond_dA, vt_cond_dA = SLA.svd(np.dot(dinvPA,dA))
            print 'Cond(dinvPA*dA):%f\n'%(s_cond_dA[0]/s_cond_dA[-1])

            dAm = dA + drhs
            u_cond_dAm, s_cond_dAm, vt_cond_dAm = SLA.svd(np.dot(dinvPA, dAm))
            print 'Cond(dinvPA*dAm):%f\n' % (s_cond_dAm[0] / s_cond_dAm[-1])

            output.s_cond_dA = s_cond_dA
            output.s_cond_dAm = s_cond_dAm

        def iter_cond_number2(A,invPA):
            # largetst eigenvalues:
            max_w_invPA_A, temp = SSLA.eigsh(A*invPA*invPA*A, k=6, tol=1e-2)
            max_w_invPA_A = max_w_invPA_A[::-1]
            # smallest eigenvalues:
                # this one does not work if det(A)==0: min_w_invPA_A, temp = SSLA.eigsh(A*invPA*invPA*A, k=6, tol=1e-2, which='SM')
                # another option.
                # min_w_invPA_A, temp = SSLA.eigsh(opt.eyelo(2*n), M=A*invPA*invPA*A, k=6, tol=1e-2)
                # min_w_invPA_A = 1 / min_w_invPA_A
            # shift the eigenvalue since A is only postive semi-definite.
            min_w_invPA_A, temp = SSLA.eigsh(A*invPA*invPA*A+1*opt.eyelo(A.shape[0]), k=6, tol=1e-2, which='SM') 
            min_w_invPA_A = min_w_invPA_A - 1
            assert(max_w_invPA_A[0]>=max_w_invPA_A[-1] and min_w_invPA_A[0]<=min_w_invPA_A[-1])
            return min_w_invPA_A, max_w_invPA_A

        if False:#check_cond_number: # this is slow for large meshes.
            min_w_invPA_A, max_w_invPA_A = iter_cond_number2(A,invPA)
            cond_invPA_A = np.sqrt(max_w_invPA_A[0] / min_w_invPA_A[0])
            output.max_w_invPA_A = max_w_invPA_A
            output.min_w_invPA_A = min_w_invPA_A
            print 'Cond(invPA*A):%f\n' % cond_invPA_A

        if False: #check_cond_number:  # this is slow for large meshes.
            Am = A + rhs
            min_w_invPA_Am, max_w_invPA_Am = iter_cond_number2(Am, invPA)
            print 'Cond(invPA*Am):%f\n' % np.sqrt(max_w_invPA_Am[0] / min_w_invPA_Am[0])
            output.max_w_invPA_Am = max_w_invPA_Am
            output.min_w_invPA_Am = min_w_invPA_Am

        if 0:
            # double check if the matrix A is positive.
            w_A = SSLA.eigsh(A,k=6,return_eigenvectors=False)
            check(w_A[0]>0)

        if verify_dense:
            dA = d(A)
            LA.norm(dA-dA.transpose()) # this is a bit slow even for small mesh
            drhs = d(rhs)
            w_dA, v_dA = SLA.eigh(dA)
            check(w_dA[0]>0)

            output.w_dA = w_dA

        print "Solving for Calderon eigenvalues...\n"
        op.tic()

        k = top_k
        if dense_solve:
            # w_A, v_A = SLA.eigh(d(A), b=d(rhs)) # eigh() requires rhs to be strictly positive.
            # w_A, v_A = SLA.eig(d(A), b=d(rhs)) # eig() can return complex eig vectors.
            # instead, use eigh with shift.
            sigma = 1e-6
            dA = d(A)
            dB = d(rhs)
            w_A, v_A = SLA.eigh(dA, b=dB+sigma*dA)
            w_A = 1/ ( 1/w_A - sigma )
            w_A = w_A[range(k)]
            v_A = v_A[:, range(k)]
            w_A_his = 'dense'
            v_A_his = 'dnese'
        else:

            # initial approximation to the k eigenvectors
            if not pre:
                print "Initialize the eigenvectors using random vectors..."
                scipy.random.seed(0)
                X = scipy.random.uniform(-1,1,size=(A.shape[0], k))
            else:
                print "Initialize the eigenvectors from precomputation..."
                X = pre.v_A
                if True:
                    # Solve for the Neumann data given the pre-Dirichlet data, this is better than interpolating the Neumann data.
                    # the following code does not work since SSLA.cg does not support b with multiple right hand side.
                    # better_neumann_init, info = SSLA.cg(L, b=(K+0.5*M).dot(pre.v_A[range(n,2*n),:]), M=invPL, tol=1e-2)
                    better_neumann_init = np.zeros([n,k])
                    B = (K + 0.5 * M).dot(pre.v_A[range(n, 2 * n), :])
                    for i in range(k):
                        better_neumann_init[:,i], _tmp_info_ = SSLA.cg(L, b=B[:,i], M=invPL, tol=1e-2)
                    X[0:n, :] = better_neumann_init
                    output.better_neumann_init = better_neumann_init
            #largest = False
            w_A, v_A, w_A_his, v_A_his = SSLA.lobpcg(A, X, B=rhs, M=invPA, maxiter=num_iter, tol=1e-3, largest=False,  verbosityLevel=1, retLambdaHistory=True, retResidualNormsHistory=True) # B=rhs,
            #if largest:
            #    w_A = w_A[::-1]
            #    v_A = v_A[:,::-1]

        outpre.v_A = v_A


        if verify_dense:
            w_dA, v_dA = SSLA.lobpcg(dA, X, maxiter=12, tol=1e-5, largest=False, verbosityLevel=1)

        print "Finished Solving for Calderon eigenvalues."
        output.timings.calderon = op.toc()

        if 0:
            plt.figure()
            plt.plot(range(k),w_A)


        if 0:
            sio.savemat('eigen',{'w_A':w_A,'v_A':v_A})

        w_S = w_A
        v_S = v_A[range(n,2*n),:]


        output.w_A = w_A
        output.v_A = v_A
        output.w_A_his = w_A_his
        output.v_A_his = v_A_his
        output.w_S = w_S
        output.v_S = v_S

    output.stdout = stdout
    output.pre = outpre


    return output #w_A, v_A[range(n,2*n),:]


import os
def save_eigen_ouput(folder,output):
    if not os.path.exists(folder):
        os.mkdir(folder)
    i = 0
    while os.path.exists(folder+'/eigen_v%d.mat'%i):
        i = i + 1
    sio.savemat(folder+'/eigen_v%d.mat'%i,output.__dict__)

def load_eigen_output(folder):
    for i in reversed(range(100)):
        #print folder + '/eigen_v%d' % i
        if os.path.isfile(folder+'/eigen_v%d.mat'%i):
            return sio.loadmat(folder + '/eigen_v%d.mat' % i)
    print "Cannot find" + folder + '/eigen_v%d.mat'


def upsample_matrix(V0,V1):
    m = V0.shape[0]
    n = V1.shape[0]
    um = np.arange(n)
    for i in range(n):
        vmin = 1e30000
        imin = 0
        for j in range(m):
            dist = np.dot( V1[i]-V0[j], V1[i]-V0[j])
            if dist < vmin:
                vmin = dist
                imin = j
        um[i] = imin
    return um

def upsample(pre0,um,m):
    pre1 = C()
    pre1.weq = pre0.weq[um]
    #pre1.v_L = pre0.v_L[um,:]
    pre1.v_invPL_L = pre0.v_invPL_L[um,:]
    pre1.v_A = pre0.v_A[np.hstack([um,um+m]),:] #since
    return pre1

def name2savefolder(model):
    name = os.path.basename(model)
    folder = './result-multires/' + name
    return folder
