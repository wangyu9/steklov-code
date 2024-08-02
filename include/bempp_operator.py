import bempp.api

from timeit import default_timer as timer

def single_layer_potential(V,F,*positional_parameters, **keyword_parameters):

    assembly_type = 'hmat' # 'dense'
    if('assembly_type' in keyword_parameters):
        assembly_type = keyword_parameters['assembly_type']

    bempp.api.boundary_operator_assembly_type = assembly_type
    bempp.api.potential_operator_assembly_type = assembly_type # not used anyway.

    vertices = V.transpose()
    elements = F.transpose()

    # FEM
    grid = bempp.api.grid_from_element_data(vertices, elements)

    #piecewise_const_space = bempp.api.function_space(grid, "DP", 0)  # A disccontinuous polynomial space of order 0
    piecewise_lin_space = bempp.api.function_space(grid, "P", 1)  # A continuous piecewise polynomial space of order 1

    slp = bempp.api.operators.boundary.laplace.single_layer(piecewise_lin_space, piecewise_lin_space,
                                                            piecewise_lin_space)
    identity = bempp.api.operators.boundary.sparse.identity(piecewise_lin_space, piecewise_lin_space,
                                                            piecewise_lin_space)

    slp_discrete = slp.weak_form()
    slp_mat_weak = bempp.api.as_matrix(slp_discrete)

    identity_discrete = identity.weak_form()
    identity_mat = bempp.api.as_matrix(identity_discrete)

    return slp_mat_weak, identity_mat

def layer_potentials(V,F):

    vertices = V.transpose()
    elements = F.transpose()

    # FEM
    grid = bempp.api.grid_from_element_data(vertices, elements)

    #piecewise_const_space = bempp.api.function_space(grid, "DP", 0)  # A disccontinuous polynomial space of order 0
    piecewise_lin_space = bempp.api.function_space(grid, "P", 1)  # A continuous piecewise polynomial space of order 1

    slp = bempp.api.operators.boundary.laplace.single_layer(piecewise_lin_space, piecewise_lin_space,
                                                            piecewise_lin_space)
    identity = bempp.api.operators.boundary.sparse.identity(piecewise_lin_space, piecewise_lin_space,
                                                            piecewise_lin_space)

    dlp = bempp.api.operators.boundary.laplace.double_layer(piecewise_lin_space, piecewise_lin_space,
                                                            piecewise_lin_space)

    slp_discrete = slp.weak_form()
    slp_mat_weak = bempp.api.as_matrix(slp_discrete)

    identity_discrete = identity.weak_form()
    identity_mat = bempp.api.as_matrix(identity_discrete)

    dlp_discrete = dlp.weak_form()
    dlp_mat_weak = bempp.api.as_matrix(dlp_discrete)

    return slp_mat_weak, dlp_mat_weak, identity_mat

def steklov_linear_system(V, F):


    vertices = V.transpose()
    elements = F.transpose()

    # FEM
    grid = bempp.api.grid_from_element_data(vertices, elements)

    #piecewise_const_space = bempp.api.function_space(grid, "DP", 0)  # A disccontinuous polynomial space of order 0
    piecewise_lin_space = bempp.api.function_space(grid, "P", 1)  # A continuous piecewise polynomial space of order 1


    slp = bempp.api.operators.boundary.laplace.single_layer(piecewise_lin_space, piecewise_lin_space,
                                                            piecewise_lin_space)


    identity = bempp.api.operators.boundary.sparse.identity(piecewise_lin_space, piecewise_lin_space,
                                                            piecewise_lin_space)

    dlp = bempp.api.operators.boundary.laplace.double_layer(piecewise_lin_space, piecewise_lin_space,
                                                            piecewise_lin_space)

    slp_discrete = slp.weak_form()

    identity_discrete = identity.weak_form()

    dlp_discrete = dlp.weak_form()

    A = slp_discrete

    B = 0.5 * identity_discrete + dlp_discrete

    return A, B


def bem_operators(V,F,*positional_parameters, **keyword_parameters):

    assembly_type = u'dense'#'hmat' # 'dense'
    if('assembly_type' in keyword_parameters):
        assembly_type = keyword_parameters['assembly_type']

    if('quadrature_order' in keyword_parameters):
        quadrature_order = keyword_parameters['quadrature_order']
        if not quadrature_order == 'default':
            print(f'quadrature_order: {quadrature_order}')
            bempp.api.global_parameters.quadrature.medium.single_order = quadrature_order
            bempp.api.global_parameters.quadrature.medium.double_order = quadrature_order

    vertices = V.transpose()
    elements = F.transpose()

    bempp.api.boundary_operator_assembly_type = assembly_type
    bempp.api.potential_operator_assembly_type = assembly_type # not used anyway.

    # FEM
    grid = bempp.api.grid_from_element_data(vertices, elements)

    #piecewise_const_space = bempp.api.function_space(grid, "DP", 0)  # A disccontinuous polynomial space of order 0
    piecewise_lin_space = bempp.api.function_space(grid, "P", 1)  # A continuous piecewise polynomial space of order 1

    slp = bempp.api.operators.boundary.laplace.single_layer(piecewise_lin_space, piecewise_lin_space,
                                                            piecewise_lin_space)
    identity = bempp.api.operators.boundary.sparse.identity(piecewise_lin_space, piecewise_lin_space,
                                                            piecewise_lin_space)

    dlp = bempp.api.operators.boundary.laplace.double_layer(piecewise_lin_space, piecewise_lin_space,
                                                            piecewise_lin_space)

    adlp = bempp.api.operators.boundary.laplace.adjoint_double_layer(piecewise_lin_space, piecewise_lin_space,
                                                            piecewise_lin_space)

    hyp = bempp.api.operators.boundary.laplace.hypersingular(piecewise_lin_space, piecewise_lin_space,
                                                             piecewise_lin_space)

    slp_discrete = slp.weak_form()
    slp_mat_weak = bempp.api.as_matrix(slp_discrete)

    identity_discrete = identity.weak_form()
    identity_mat = bempp.api.as_matrix(identity_discrete)

    dlp_discrete = dlp.weak_form()
    dlp_mat_weak = bempp.api.as_matrix(dlp_discrete)

    adlp_discrete = adlp.weak_form()
    adlp_mat_weak = bempp.api.as_matrix(adlp_discrete)

    hyp_discrete = hyp.weak_form()
    hyp_mat_weak = bempp.api.as_matrix(hyp_discrete)

    return slp_mat_weak, dlp_mat_weak, identity_mat, adlp_mat_weak, hyp_mat_weak

import numpy as np
import scipy.linalg as SLA

def steklov_operator(V,F,*positional_parameters, **keyword_parameters):

    assembly_type = u'dense'#'#'hmat' # 'dense'
    quadrature_order = 'default'
    custom_parameter = False
    sparse_mass = False
    if('assembly_type' in keyword_parameters):
        assembly_type = keyword_parameters['assembly_type']
        custom_parameter = True
    if('quadrature_order' in keyword_parameters):
        quadrature_order = keyword_parameters['quadrature_order']
        custom_parameter = True
    if('sparse_mass' in keyword_parameters):
        sparse_mass = keyword_parameters['sparse_mass']

    if custom_parameter:
        L, K, M_sp, Ka, H = bem_operators(V, F, assembly_type=assembly_type, quadrature_order=quadrature_order)
    else:
        L, K, M_sp, Ka, H = bem_operators(V, F)
    invL = SLA.inv(L)
    M = M_sp.toarray()
    S = H + np.dot(0.5 * M + K.transpose(), np.dot(invL, 0.5 * M + K))

    if sparse_mass:
        return S, M_sp
    else:
        return S, M

def single_double_layer_operators(V,F):

    vertices = V.transpose()
    elements = F.transpose()

    # FEM
    grid = bempp.api.grid_from_element_data(vertices, elements)

    #piecewise_const_space = bempp.api.function_space(grid, "DP", 0)  # A disccontinuous polynomial space of order 0
    piecewise_lin_space = bempp.api.function_space(grid, "P", 1)  # A continuous piecewise polynomial space of order 1

    slp = bempp.api.operators.boundary.laplace.single_layer(piecewise_lin_space, piecewise_lin_space,
                                                            piecewise_lin_space)
    identity = bempp.api.operators.boundary.sparse.identity(piecewise_lin_space, piecewise_lin_space,
                                                            piecewise_lin_space)
    dlp = bempp.api.operators.boundary.laplace.double_layer(piecewise_lin_space, piecewise_lin_space,
                                                            piecewise_lin_space)

    slp_discrete = slp.weak_form()
    #slp_mat_weak = bempp.api.as_matrix(slp_discrete)

    identity_discrete = identity.weak_form()
    #identity_mat = bempp.api.as_matrix(identity_discrete)

    dlp_discrete = dlp.weak_form()
    #dlp_mat_weak = bempp.api.as_matrix(dlp_discrete)

    return slp_discrete, dlp_discrete, identity_discrete

def boundary_operators(V,F,*positional_parameters, **keyword_parameters):

    assembly_type = u'dense' # 'hmat' # 'dense'
    if('assembly_type' in keyword_parameters):
        assembly_type = keyword_parameters['assembly_type']

    vertices = V.transpose()
    elements = F.transpose()

    bempp.api.boundary_operator_assembly_type = assembly_type
    bempp.api.potential_operator_assembly_type = assembly_type # not used anyway.
    # FEM
    grid = bempp.api.grid_from_element_data(vertices, elements)

    #piecewise_const_space = bempp.api.function_space(grid, "DP", 0)  # A disccontinuous polynomial space of order 0
    piecewise_lin_space = bempp.api.function_space(grid, "P", 1)  # A continuous piecewise polynomial space of order 1

    slp = bempp.api.operators.boundary.laplace.single_layer(piecewise_lin_space, piecewise_lin_space,
                                                            piecewise_lin_space)
    identity = bempp.api.operators.boundary.sparse.identity(piecewise_lin_space, piecewise_lin_space,
                                                            piecewise_lin_space)

    dlp = bempp.api.operators.boundary.laplace.double_layer(piecewise_lin_space, piecewise_lin_space,
                                                            piecewise_lin_space)

    adlp = bempp.api.operators.boundary.laplace.adjoint_double_layer(piecewise_lin_space, piecewise_lin_space,
                                                            piecewise_lin_space)

    hyp = bempp.api.operators.boundary.laplace.hypersingular(piecewise_lin_space, piecewise_lin_space,
                                                             piecewise_lin_space)

    slp_discrete = slp.weak_form()

    identity_discrete = identity.weak_form()

    dlp_discrete = dlp.weak_form()

    adlp_discrete = adlp.weak_form()

    hyp_discrete = hyp.weak_form()

    return slp_discrete, dlp_discrete, adlp_discrete, hyp_discrete, identity_discrete

def symmetrized_boundary_operators(V,F,*positional_parameters, **keyword_parameters):

    assembly_type = 'hmat' # 'dense'
    eps = 1e-3
    if('assembly_type' in keyword_parameters):
        assembly_type = keyword_parameters['assembly_type']
    if('eps' in keyword_parameters):
        eps = keyword_parameters['eps']

    print('eps: %f\n' % eps)

    vertices = V.transpose()
    elements = F.transpose()

    bempp.api.boundary_operator_assembly_type = assembly_type
    bempp.api.potential_operator_assembly_type = assembly_type # not used anyway.

    # bempp.api.global_parameters.hmat.eps = eps
    assert(eps==1e-3) # default value. TODO: set eps using the new API. 
    
    # FEM
    # grid = bempp.api.grid_from_element_data(vertices, elements)
    grid = bempp.api.Grid(vertices, elements)
    
    #piecewise_const_space = bempp.api.function_space(grid, "DP", 0)  # A disccontinuous polynomial space of order 0
    piecewise_lin_space = bempp.api.function_space(grid, "P", 1)  # A continuous piecewise polynomial space of order 1

    slp = bempp.api.operators.boundary.laplace.single_layer(piecewise_lin_space, piecewise_lin_space,
                                                            piecewise_lin_space)
    identity = bempp.api.operators.boundary.sparse.identity(piecewise_lin_space, piecewise_lin_space,
                                                            piecewise_lin_space)

    dlp = bempp.api.operators.boundary.laplace.double_layer(piecewise_lin_space, piecewise_lin_space,
                                                            piecewise_lin_space)

    #adlp = bempp.api.operators.boundary.laplace.adjoint_double_layer(piecewise_lin_space, piecewise_lin_space,
    #                                                        piecewise_lin_space)

    hyp = bempp.api.operators.boundary.laplace.hypersingular(piecewise_lin_space, piecewise_lin_space,
                                                             piecewise_lin_space)

    slp_discrete = 0.5*(slp.weak_form()+slp._transpose(piecewise_lin_space).weak_form())

    identity_discrete = identity.weak_form()

    dlp_discrete = dlp.weak_form()

    adlp_discrete = dlp._transpose(piecewise_lin_space).weak_form()

    hyp_discrete = 0.5*(hyp.weak_form()+hyp._transpose(piecewise_lin_space).weak_form())

    return slp_discrete, dlp_discrete, adlp_discrete, hyp_discrete, identity_discrete

def transposed_boundary_operators(V,F,*positional_parameters, **keyword_parameters):

    assembly_type = 'hmat' # 'dense'
    if('assembly_type' in keyword_parameters):
        assembly_type = keyword_parameters['assembly_type']

    vertices = V.transpose()
    elements = F.transpose()

    bempp.api.boundary_operator_assembly_type = assembly_type
    bempp.api.potential_operator_assembly_type = assembly_type # not used anyway.
    # FEM
    grid = bempp.api.grid_from_element_data(vertices, elements)

    #piecewise_const_space = bempp.api.function_space(grid, "DP", 0)  # A disccontinuous polynomial space of order 0
    piecewise_lin_space = bempp.api.function_space(grid, "P", 1)  # A continuous piecewise polynomial space of order 1

    slp = bempp.api.operators.boundary.laplace.single_layer(piecewise_lin_space, piecewise_lin_space,
                                                            piecewise_lin_space)

    hyp = bempp.api.operators.boundary.laplace.hypersingular(piecewise_lin_space, piecewise_lin_space,
                                                             piecewise_lin_space)

    slpt_discrete = slp.transpose(piecewise_lin_space).weak_form()

    hypt_discrete = hyp.transpose(piecewise_lin_space).weak_form()

    return slpt_discrete, hypt_discrete

def time():
    return timer()

import time

lock_for_tictoc = False

def tic():
    # http://stackoverflow.com/questions/5849800/tic-toc-functions-analog-in-python
    #Homemade version of matlab tic and toc functions
    global startTime_for_tictoc
    global lock_for_tictoc
    assert(lock_for_tictoc==False)
    startTime_for_tictoc = time.time()
    lock_for_tictoc = True


def toc(silent=False):
    global endTime_for_tictoc
    global duration_for_tic_toc
    global lock_for_tictoc
    assert(lock_for_tictoc==True)
    endTime_for_tictoc = time.time()
    duration_for_tic_toc = str( endTime_for_tictoc- startTime_for_tictoc)
    if not silent:
        if 'startTime_for_tictoc' in globals():
            print("Elapsed time is " + duration_for_tic_toc + " seconds.")
        else:
            print("Toc: start time not set")
    lock_for_tictoc = False
    return duration_for_tic_toc

def duration():
    return duration_for_tic_toc

def ieigh(A,M):
    #from scipy.linalg import
    w,v = SLA.eigh(A,b=M)
    return 1/w[::-1], v[:,::-1]

def compute_weights(S,B,BC):
    n = S.shape[0]

    nB = list(set(range(n)) - set(B))

    m = BC.shape[1]
    W_s = np.zeros([n, m])

    W_s[B, :] = BC

    Sbb = S[B][:, B]
    Sbn = S[B][:, nB]
    Snb = S[nB][:, B]
    Snn = S[nB][:, nB]

    W_s[nB, :] = - SLA.solve(Snn, np.dot(Snb, BC))

    return W_s

def compute_weights_sparse(S,B,BC):
    n = S.shape[0]

    nB = list(set(range(n)) - set(B))

    m = BC.shape[1]
    W_s = np.zeros([n, m])

    W_s[B, :] = BC

    Sbb = S[B][:, B]
    Sbn = S[B][:, nB]
    Snb = S[nB][:, B]
    Snn = S[nB][:, nB]

    W_s[nB, :] = - SLA.solve(Snn, np.dot(Snb, BC))

    return W_s


def heat_kernel_signature(VV1,DD1,Ts,k,p):

    N = len(Ts)
    Hks = np.zeros(N)

    for ii in range(N):
        t = Ts[ii]
        Hk = np.dot( VV1[p,range(k)], np.dot(np.diag(np.exp(-t*DD1[range(k)])), VV1[p,range(k)].transpose()))
        # total HKS
        tHk = ( VV1[:,range(k)] * np.dot( VV1[:,range(k)], np.diag(np.exp(-t*DD1[range(k)]))) ).sum()
        Hks[ii] = Hk

    return Hks

def total_heat_kernel_signature(VV1,DD1,Ts,k):

    N = len(Ts)
    tHks = np.zeros(N)

    for ii in range(N):
        t = Ts[ii]
        # total HKS
        tHk = ( VV1[:,range(k)] * np.dot( VV1[:,range(k)], np.diag(np.exp(-t*DD1[range(k)]))) ).sum()
        tHks[ii] = tHk

    return tHks
