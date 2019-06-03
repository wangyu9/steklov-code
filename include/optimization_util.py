from scipy import linalg as SLA
from scipy.sparse import linalg as SSLA
import scipy.sparse as sp
import numpy as np


def QuadOptSingle(A,b,Aeq,beq):

    n = A.shape[0]
    m = Aeq.shape[0]
    Aeqt = Aeq.transpose()

    def mv(v):
        assert(v.shape[0]==m+n)
        u = v[range(n)]
        l = v[range(n,m+n)]
        r = np.zeros(m+n)
        r[range(n)] = A.matvec(u) + Aeqt.dot(l)
        r[range(n,m+n)] = Aeq.dot(u)
        return r

    Linear = SSLA.LinearOperator( shape=(m+n,m+n), matvec=mv, dtype=float)
    rhs = np.zeros(m+n)
    rhs[range(n)] = b
    rhs[range(n,m+n)] = beq

    result = SSLA.cgs(Linear,rhs,tol=1e-3)
    if result[1]==0:
        x = result[0]
    else:
        print 'cg solver does not converge\n'

    return x

def QuadOptKnownSingle(A,b,known,knownY):

    n = A.shape[0]
    m = known.shape[0]
    unknown = list(set(range(n))-set(known))
    bk = b[known]
    bu = b[unknown]

    def mv(v):
        assert(v.shape[0]==n-m)
        # this trick allows me to compute A_uu * v
        z = np.zeros(n)
        z[unknown] = v
        r = A.matvec(z)
        return r[unknown]

    Linear = SSLA.LinearOperator( shape=(n-m,n-m), matvec=mv, dtype=float)
    # this trick allows me to compute A_uk * x_k
    zk = np.zeros(n)
    zk[known] = knownY
    rhs = bu - A.matvec(zk)[unknown]

    x = np.zeros(n)
    x[known] = knownY
    # cgs solver for non-symmetric system.
    result = SSLA.cgs(Linear, rhs, tol=1e-18)
    if result[1]==0:
        x[unknown] = result[0]
    else:
        print 'cg solver does not converge\n'

    return x

def QuadOpt(A,B,Aeq=[],Beq=[]):
    if not Aeq:
        # Aeq == []
        Aeq = np.zeros([0,A.shape[1]])
        assert(not Beq)
        Beq = np.zeros([0,B.shape[1]])
    if(B.ndim==1):
        return QuadOptSingle(A,B,Aeq,Beq)
    h = B.shape[1]
    n = A.shape[1]
    assert(h==Beq.shape[1])
    X = np.zeros([n,h])
    for ii in range(h):
        X[:,ii] = QuadOptSingle(A,B[:,ii],Aeq,Beq[:,ii])
    return X

def QuadOptDense(A, B, known, knownY):
    n = A.shape[1]
    h = B.shape[1]
    unknown = list(set(range(n)) - set(known))
    X = np.zeros([n,h])
    X[known,:] = knownY
    X[unknown,:] = - SLA.solve( A[unknown][:,unknown], np.dot(A[unknown][:,known],knownY) )
    return X

def QuadOptKnown(A,B,known,knownY):
    if(B.ndim==1):
        return QuadOptKnownSingle(A,B,known,knownY)
    h = B.shape[1]
    n = A.shape[1]
    assert(h==knownY.shape[1])
    X = np.zeros([n,h])
    for ii in range(h):
        X[:,ii] = QuadOptKnownSingle(A,B[:,ii],known,knownY[:,ii])
    return X

def InvQuadOpt(H,Q,T,V,Aeq,Beq):
    n = H.shape[0]
    m = Aeq.shape[0]
    h = Beq.shape[1]
    assert(n==H.shape[1])
    assert(n==Q.shape[0])
    ix = range(n)
    iy = range(n, 2 * n)

    def mv(v):
        x = v[ix]
        y = v[iy]
        r = np.zeros(2*n)
        r[ix] = H.matvec(x) + T.matvec(y)
        r[iy] = Q.matvec(x) - V.matvec(y)
        return r

    A = SSLA.LinearOperator(shape=(2*n,2*n), matvec=mv)
    BB = np.zeros([n,h])

    AAeq = sp.coo_matrix((Aeq.data, (Aeq.row, Aeq.col)), shape=(m, 2*n))

    result = QuadOpt(A, BB, AAeq, Beq)

    X = result[ix,:]
    Y = result[iy,:]

    return X

def transpose(A):
    return SSLA.LinearOperator(shape=(A.shape[1],A.shape[0]),matvec=A.rmatvec,rmatvec=A.matvec, dtype=float)

def vstack_core(blocks):
    size = len(blocks)
    cols = blocks[0].shape[1]
    r = range(size)
    starts = range(size)
    ends = range(size)
    count_sum = 0
    for ii in range(size):
        assert(blocks[0].shape[1]==cols)
        r[ii] = blocks[ii].shape[0]
        count_sum = count_sum + r[ii]
        ends[ii] = count_sum
        if ii+1<size:
            starts[ii+1] = count_sum
    rows = count_sum

    def mv(v):
        result = np.zeros(rows)
        for ii in range(size):
            result[range(starts[ii],ends[ii])] = blocks[ii].matvec(v)
        return result

    return SSLA.LinearOperator(shape=(rows,cols),matvec=mv,dtype=float)

def hstack_core(blocks):
    size = len(blocks)
    rows = blocks[0].shape[0]
    r = range(size)
    starts = range(size)
    ends = range(size)
    count_sum = 0
    for ii in range(size):
        assert(blocks[0].shape[0]==rows)
        r[ii] = blocks[ii].shape[1]
        count_sum = count_sum + r[ii]
        ends[ii] = count_sum
        if ii+1<size:
            starts[ii+1] = count_sum
    cols = count_sum

    def mv(v):
        result = np.zeros(rows)
        for ii in range(size):
            result = result + blocks[ii].matvec(v[range(starts[ii],ends[ii])])
        return result

    return SSLA.LinearOperator(shape=(rows,cols),matvec=mv,dtype=float)

def naive_matmat(matvec):
    def mm(M):
        for i in range(M.shape[1]):
            r = matvec(M[:,i])
            if i==0:
                R = np.zeros([r.shape[0],M.shape[1]])
            R[:,i] = r
        return R
    return mm

def vstack(blocks):
    vlo = vstack_core(blocks)
    trans_blocks = [transpose(A) for A in blocks]
    hlo = hstack_core(trans_blocks)
    vlo.rmatvec = hlo.matvec
    return vlo

def hstack(blocks):
    hlo = hstack_core(blocks)
    trans_blocks = [transpose(A) for A in blocks]
    vlo = vstack_core(trans_blocks)
    hlo.rmatvec = vlo.matvec
    return hlo

def inv_sparse(M,Mt):
    assert(M.shape[0]==M.shape[1])
    assert(Mt.shape[0] == Mt.shape[1])
    assert(M.shape[0] == Mt.shape[0])
    def mv(v):
        return SSLA.spsolve(M, v)
    def rmv(v):
        return SSLA.spsolve(Mt, v)
    return SSLA.LinearOperator(shape=M.shape,matvec=mv,rmatvec=rmv)

'''
# this compute A\B = inv(A) * B
def mldivide_core(A,B):
    assert(A.shape[0]==A.shape[1])
    # assume A is a sparse matrix
    def mv(v):
        return SSLA.spsolve( A, B.matvec(v))
    return SSLA.LinearOperator(shape=B.shape,matvec=mv,dtype=float)

# this compute A/B = A * inv(B)
def mrdivide_core(A,B):
    assert(B.shape[0]==B.shape[1])
    # assume A is a sparse matrix
    def mv(v):
        return A.matvec( SSLA.spsolve( B, v) )
    return SSLA.LinearOperator(shape=A.shape,matvec=mv,dtype=float)

def mldivide(A,B):
    leftlo =  mldivide_core(A,B)
    At = transpose(A)
    Bt = transpose(B)
    rightlo = mrdivide_core(Bt,At)
    leftlo.rmatvec = rightlo.matvec
    return leftlo

def mrdivide(A,B):
    rightlo =  mrdivide_core(A,B)
    At = transpose(A)
    Bt = transpose(B)
    leftlo = mldivide_core(Bt,At)
    rightlo.rmatvec = leftlo.matvec
    return rightlo
'''

def array2rlo_core(a):
    def mv(v):
        return np.dot(a,v)
    return SSLA.LinearOperator(shape=(1,a.shape[0]),matvec=mv,dtype=float)

def array2clo_core(a):
    def mv(v):
        # assert(v.shape==(1,1))
        return v* a
    return SSLA.LinearOperator(shape=(a.shape[0],1),matvec=mv,dtype=float)

def array2rlo(a):
    rowlo = array2rlo_core(a)
    collo = array2clo_core(a)
    rowlo.rmatvec = collo.matvec
    return rowlo

def array2clo(a):
    collo = array2clo_core(a)
    rowlo = array2rlo_core(a)
    collo.rmatvec = rowlo.matvec
    return collo

# sparse matrix to linear operator
def s2lo_core(M):
    def mv(v):
        return M.dot(v)
    return SSLA.LinearOperator(shape=M.shape,matvec=mv,dtype=float)

def s2lo(M,Mt):
    lo = s2lo_core(M)
    tlo = s2lo_core(Mt)
    lo.rmatvec = tlo.matvec
    return lo

def diag2lo(w):
    n = w.shape[0]
    def mv(v):
        return v*w
    return SSLA.LinearOperator(shape=(n,n),matvec=mv,rmatvec=mv,matmat=naive_matmat(mv),dtype=float)

# return a zero linear operator.
def zlo(size):
    def mv(v):
        return np.zeros(size[0])
    def rmv(v):
        return np.zeros(size[1])
    return SSLA.LinearOperator(shape=size,matvec=mv,rmatvec=rmv,dtype=float)

def bem2lo(M):
    return SSLA.LinearOperator(shape=M.shape, matvec=M.matvec, rmatvec=M.rmatvec, matmat=M.matmat, dtype=float)

def eyelo(size):
    def mv(v):
        return v
    return SSLA.LinearOperator(shape=(size,size), matvec=mv, rmatvec=mv, dtype=float)

def adjoint(A):
    return SSLA.LinearOperator(shape=(A.shape[1],A.shape[0]),matvec=A.rmatvec,rmatvec=A.matvec, dtype=float)

def adjoint_average(A):
    size = A.shape[0]
    assert(size==A.shape[1])
    def mv(v):
        return 0.5*(A.matvec(v)+A.rmatvec(v))
    return SSLA.LinearOperator(shape=(size,size), matvec=mv, rmatvec=mv, dtype=float)

# this solve the saddle point system as describe in page 319 in the book [Oalf 2008]
# (A    -B    (u1  =  (f1
#  B^t   D)    u2)      f2)
def saddle_transform(A,BT,invPA):
    n = A.shape[0]
    I = eyelo(n)
    T1 = hstack([ A*invPA-I, zlo(n)])
    T2 = hstack([ -BT*invPA, I])
    T = vstack([T1,T2])
    return T

def saddle_system_matrix(A, B, BT, D, invPA):
    n = A.shape[0]
    I = eyelo(n)
    M1 = hstack([ A*invPA*A - A, (I-A*invPA)*B])
    M2 = hstack([ BT*(I-invPA*A), D+BT*invPA*B])
    M = vstack([M1, M2])
    return M

def saddle_system_matrix2(A, B, BT, C, invPA):
    n = A.shape[0]
    I = eyelo(n)
    M1 = hstack([ A*invPA*A - A, (I-A*invPA)*B])
    M2 = hstack([ BT*(I-invPA*A), D+BT*invPA*B])
    M = vstack([M1, M2])
    return M

def saddle_rhs_matrix(M):
    n = M.shape[0]
    Z = zlo([n,n])
    R1 = hstack([Z,Z])
    R2 = hstack([Z,M])
    R = vstack([R1,R2])
    return R

def blkdiag(blocks):
    size = len(blocks)
    assert (size==2)
    M0 = blocks[0]
    M1 = blocks[1]
    n0 = M0.shape[0]
    n1 = M1.shape[1]
    assert (M0.shape[1]==n0)
    assert (M1.shape[1]==n1)
    R1 = hstack([M0,zlo([n0,n1])])
    R2 = hstack([zlo([n1,n0]),M1])
    R = vstack([R1,R2])
    return R

'''
# finishing this function later.
def blkdiag(blocks):
    size = len(blocks)
    dims0 = [A.shape[0] for A in blocks]
    dims1 = [A.shape[1] for A in blocks]
'''
