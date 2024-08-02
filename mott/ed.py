###################################################
#      Exact-diagonalization for impurity model
#      Author: Tsung-Han Lee
#      email: henhans74716@gmail.com  
###################################################
import numpy as np
from scipy.sparse import csr_matrix, identity
from scipy.sparse.linalg import eigsh, expm
from scipy.linalg import block_diag
from itertools import product
#import primme
import numba
from numba import jit, prange
from dos import *

def build_fermion_op(no):
    '''
    Build fermionic operators for each spin+orbitals, alpha, with size 
    of Hilberspace, A and B.
    Input:
        no: number of orbital
    Output:
        FH_list: a list of fermionic operator <A|[f^dagger_alpha]|B>
    '''
    
    hsize = 2**no # size of Hilbert space
    strb = '{0:0'+str(no)+'b}' # string to save binary configuration
    FH_list = []
    #print 'Hibert space size:', hsize
    for o in range(no): #calculate FH for each orbital
        #print 'building FH for orbital:', o
        row = [] #storing row index for 
        col = [] #storing col index for
        data = [] #storing data for (row, col)
        for b in range(hsize): # col |B>
            config_b = strb.format(b) # cofiguration correspond to B
            #print config_b #print binary configuration
            #print config_b[5]
            if config_b[o] == '0':
                col.append(b)
                config_a = ''
                #calculate the corresponding <A|
                for i in range(no):
                    if i == o:
                        config_a += '1'
                    else:
                        config_a += config_b[i]
                a = int(config_a,2)
                row.append(a)
                #calculate the exponent for minus sign
                expo = 0
                for i in range(o):
                    expo += int(config_b[i])
                data.append((-1.)**(expo))
                #print config_a
                #print config_b
                #print (-1.)**(expo), expo
        #print '<|=',row
        #print '|>=',col
        #print 'data=',data
        #assign values into sparse matrix
        FH = csr_matrix((data, (row, col)), shape=(hsize, hsize), dtype=float)
        #print FH.toarray()
        FH_list.append(FH)
    return FH_list

def build_Hemb_matrix(h1, V, eb, V2E, FH_list):
    '''
    Build Hemb matrix.
    Input:
        V: hybridization matrix
        h1: local one-body matrix
        eb: bath one-body levels
        V2E: local two-body interaction if type(V2E)==list recompute U_matrix, if type(V2E)==sparse.matrix add it to Hemb
        FH_list: fermionic operator list
    Return:
        Hemb: embedding Hamiltonian
    '''
    no = h1.shape[0] + eb.shape[0]
    ni = h1.shape[0]
    nb = eb.shape[0]
    hsize = 2**no
    Hemb = csr_matrix((hsize,hsize), dtype=complex)
    H1mat = csr_matrix((hsize,hsize), dtype=complex)
    Vmat = csr_matrix((hsize,hsize), dtype=complex)
    Vconjmat = csr_matrix((hsize,hsize), dtype=complex)
    ebmat = csr_matrix((hsize,hsize), dtype=complex)
    
    #build local one-body part
    for i in range(ni):
        for j in range(ni):
            #print( i, j)
            Hemb += h1[i,j]*FH_list[i].dot( FH_list[j].getH() )
            H1mat+= h1[i,j]*FH_list[i].dot( FH_list[j].getH() )
    #build hybridization part
    for i in range(ni):
        for j in range(nb):
            Hemb += V[j,i]*FH_list[i].dot( FH_list[ni+j].getH() )
            Hemb += V[j,i].conj()*FH_list[ni+j].dot( FH_list[i].getH() )
            Vmat += V[j,i]*FH_list[i].dot( FH_list[ni+j].getH() )
            Vconjmat += V[j,i].conj()*FH_list[ni+j].dot( FH_list[i].getH() )
    #build bath part
    for i in range(nb):
            Hemb += eb[i]*FH_list[ni+i].dot( FH_list[ni+i].getH() )
            ebmat += eb[i]*FH_list[ni+i].dot( FH_list[ni+i].getH() )
    #build local two-body part
    U2loc = csr_matrix((hsize,hsize), dtype=complex)
    if type(V2E) == np.ndarray:
        for i in range(ni):
            for j in range(ni):
                for k in range(ni):
                    for l in range(ni):
                        #print i,j,k,l
                        if np.abs(V2E[i,j,k,l]) > 1e-5:
                            Hemb += 0.5*V2E[i,j,k,l]*FH_list[i].dot( FH_list[k].dot( FH_list[l].getH().dot(FH_list[j].getH() ) ) )
                            U2loc += 0.5*V2E[i,j,k,l]*FH_list[i].dot( FH_list[k].dot( FH_list[l].getH().dot(FH_list[j].getH() ) ) )
    elif type(V2E) == csr_matrix:
        Hemb += V2E
        U2loc = V2E
    else:
        raise Exception("V2E has to be either numpy.ndarray or sparse.csr_matrix!")

    return Hemb, U2loc

def build_U2loc_matrix(V2E, eb, FH_list):
    '''
    Build Hemb matrix.
    Input:
        V2E: local two-body interaction if type(V2E)==list recompute U_matrix
        FH_list: fermion operator list
    Return:
        U2loc: Two-body interaction Hamiltonian
    '''
    
    no = V2E.shape[0] + eb.shape[0]
    ni = V2E.shape[0]
    hsize = 2**no
    #build local two-body part
    U2loc = csr_matrix((hsize,hsize), dtype=complex)
    for i in range(ni):
        for j in range(ni):
            for k in range(ni):
                for l in range(ni):
                    #print i,j,k,l
                    U2loc += 0.5*V2E[i,j,k,l]*FH_list[i].dot( FH_list[k].dot( FH_list[l].getH().dot(FH_list[j].getH() ) ) )

    return U2loc

def calc_density_matrix_thermal(FH_list, T, evals, evecs):
    '''
    calculate density matrix.
    Input:
        FH_list: fermion operator list
        vec: ground state eigenvector
    Return:
        dm: density matrix
    '''
    exp_bE = np.diag(np.exp(-evals/T))
    Z = np.trace(exp_bE)

    no = len(FH_list)
    dm = np.zeros((no,no),dtype=complex)
    for i in range(no):
        for j in range(no):
            dm[i,j] = np.trace( evecs.dot(exp_bE).dot(evecs.conj().T).dot( (FH_list[i].dot( FH_list[j].getH() )).todense() ) )/Z
    return dm

def calc_E2loc_thermal(U2loc,rho):
    '''
    calculate density matrix.
    Input:
        U2loc: local two-body interaction operator
        vec: ground state eigenvector
    Return:
        local two-body ground state energy
    '''
    return np.trace(rho.dot( U2loc.todense() ) )/np.trace(rho) 

def calc_double_occ_thermal(idx,FH_list, T, evals, evecs):
    '''
    calculate double occupancy at orbital idx.
    Input:
        idx: index for the orbital (also idx+1) where the double occupancy is calculated.
        FH_list: fermion operator list.
        vec: ground state eigenvector.
    Return:
        double occupancy
    '''
    exp_bE = np.diag(np.exp(-evals/T))
    Z = np.trace(exp_bE)
    return np.trace( evecs.dot(exp_bE).dot(evecs.conj().T).dot( ( FH_list[idx].dot( FH_list[idx].getH().dot( 
             FH_list[idx+1].dot( FH_list[idx+1].getH() ) ) ) ).todense() ) )/Z

def solve_Hemb_thermal(T, h1, V, eb, V2E, FH_list, verbose=0):
    '''
    solve embeding impurity Hamiltonian.
    Input:
        V: hybridization matrix
        h1: local one-body matrix
        eb: bath one-body 
        V2E: local two-body interaction if type(V2E)==list recompute U_matrix, if type(V2E)==sparse.matrix add it to Hemb
        FH_list: fermion operator list
        verbose: printing message
    Return:
        density matrix
        evals
        evecs
    '''
    no = h1.shape[0] + eb.shape[0]

    hsize = 2**no
    #print '# of orbitals=',no

    #print('h1=')
    #print(h1)
    #print('V=')
    #print(V)
    #print('eb=')
    #print(eb)

    Hemb, U2loc = build_Hemb_matrix(h1, V, eb, V2E, FH_list)

    Hemb_dense = Hemb.todense()
    # using scipy.linalg.eigh
    from scipy.linalg import eigh
    evals, evecs = eigh(Hemb_dense)
    e0 = evals[0]
    evals = evals - e0

    print('Egs=', e0)
    print('evals=',evals[:10])

    dm = calc_density_matrix_thermal(FH_list, T, evals, evecs)

    docc = calc_double_occ_thermal(0,FH_list, T, evals, evecs)

    return dm, evals, evecs, docc

def operators_to_eigenbasis(op_vec, U):
    dop_vec = []
    for op in op_vec:
        dop = np.mat(U).H * op * np.mat(U)
        dop_vec.append(dop)

    return dop_vec

def compute_GomF_thermal(T, omFs, evals, evecs, FH_list_dense):
    # -- Components of the Lehman expression
    dE = - evals[:, None] + evals[None, :]
    exp_bE = np.exp(- evals/T)
    M = exp_bE[:, None] + exp_bE[None, :]

    inv_freq = 1j*omFs[:, None, None] - dE[None, :, :]
    nonzero_idx = np.nonzero(inv_freq)
    # -- Only eval for non-zero values
    freq = np.zeros_like(inv_freq)
    freq[nonzero_idx] = inv_freq[nonzero_idx]**(-1)

    op1_eig, op2_eig = operators_to_eigenbasis([FH_list_dense[0].conj().T, FH_list_dense[0]], evecs)

    # -- Compute Lehman sum for all operator combinations
    Gw = np.zeros((len(omFs)), dtype=complex)
    Gw = np.einsum('nm,mn,nm,znm->z', op1_eig, op2_eig, M, freq)
    Gw /= np.sum(exp_bE)
    return Gw

def compute_Gw_thermal(T, oms, eta, evals, evecs, FH_list_dense):
    # -- Components of the Lehman expression
    dE = - evals[:, None] + evals[None, :]
    exp_bE = np.exp(- evals/T)
    M = exp_bE[:, None] + exp_bE[None, :]

    inv_freq = (oms+1j*eta)[:, None, None] - dE[None, :, :]
    nonzero_idx = np.nonzero(inv_freq)
    # -- Only eval for non-zero values
    freq = np.zeros_like(inv_freq)
    freq[nonzero_idx] = inv_freq[nonzero_idx]**(-1)

    op1_eig, op2_eig = operators_to_eigenbasis([FH_list_dense[0].conj().T, FH_list_dense[0]], evecs)

    # -- Compute Lehman sum for all operator combinations
    Gw = np.zeros((len(oms)), dtype=complex)
    Gw = np.einsum('nm,mn,nm,znm->z', op1_eig, op2_eig, M, freq)
    Gw /= np.sum(exp_bE)
    return Gw


@jit(nopython=True)
def compute_Gw(oms, eta, evals, evecs, FH_list_dense):
    Gw = np.zeros(len(oms),dtype=numba.complex128)
    for iom,om in enumerate(oms):
        print(iom)
        tmp = 0.0
        for i,ei in enumerate(evals):
            ifdg = evecs[:,i].conj().T.dot(FH_list_dense[0]).dot(evecs[:,0])
            gfi = evecs[:,0].conj().T.dot(FH_list_dense[0].conj().T).dot(evecs[:,i])
            tmp += ifdg*gfi/(ei-evals[0]-om-1j*eta)
            ifg = evecs[:,i].conj().T.dot(FH_list_dense[0].conj().T).dot(evecs[:,0])
            gfdi = evecs[:,0].conj().T.dot(FH_list_dense[0]).dot(evecs[:,i])
            tmp += ifg*gfdi/(evals[0]-ei-om-1j*eta)
        Gw[iom] = tmp
    return Gw

@jit(nopython=True, parallel=True)
def compute_GomF(omFs, evals, evecs, FH_list_dense, deg):#gs_idx=[0]):
    Gw = np.zeros(len(omFs),dtype=numba.complex128)
    #Gw = np.zeros(len(omFs),dtype=complex128)
    for iomF in prange(len(omFs)):
        omF = omFs[iomF]
        #print(iomF)
        tmp = 0.0
        for i,ei in enumerate(evals):
            ifdg = evecs[:,i].conj().T.dot(FH_list_dense[0]).dot(evecs[:,:deg])
            gfi = evecs[:,:deg].conj().T.dot(FH_list_dense[0].conj().T).dot(evecs[:,i])
            tmp += ifdg.dot(gfi)/(ei-evals[0]-1j*omF)
            ifg = evecs[:,i].conj().T.dot(FH_list_dense[0].conj().T).dot(evecs[:,:deg])
            gfdi = evecs[:,:deg].conj().T.dot(FH_list_dense[0]).dot(evecs[:,i])
            tmp += ifg.dot(gfdi)/(evals[0]-ei-1j*omF)
        Gw[iomF] = -tmp/deg
    return Gw

def compute_GlattomF_semcircle(omFs,Sig,mu):
    Glattw = np.zeros((len(omFs)),dtype=np.complex128)
    t = 0.5 # half-bandwidth set to 1
    es = np.linspace(-2*t,2*t,2000)
    de = es[1] - es[0]
    dose = semicircle(es,t)
    for iw in range(len(omFs)):
        tmp = 0j
        for ie, e in enumerate(es):
            tmp += dose[ie]/(omFs[iw]+mu-e-Sig[iw])
        Glattw[iw] = tmp*de
    return Glattw

def compute_Glattw_semcircle(oms,eta,Sig,mu):
    Glattw = np.zeros((len(oms)),dtype=np.complex128)
    t = 0.5 # half-bandwidth set to 1
    es = np.linspace(-2*t,2*t,2000)
    de = es[1] - es[0]
    dose = semicircle(es,t)
    for iw in range(len(oms)):
        tmp = 0j
        for ie, e in enumerate(es):
            tmp += dose[ie]/(oms[iw]+1j*eta+mu-e-Sig[iw])
        Glattw[iw] = tmp*de
    return Glattw

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    np.set_printoptions(suppress=True, precision=8)

    U = 0.0
    #
    T = 0.005
    #
    h1 = np.array([[-U/2., 0.0],
                   [ 0.0,-U/2.]])
    #
    #eb = np.array([-0.1,-0.1, 0.0, 0.0, 0.1, 0.1])
    eb = np.array([ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    #eb = np.array([ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    #eb = np.array([ 0.0, 0.0, 0.0, 0.0])
    #
    #V = np.array([[ 0.1],
    #              [ 0.1],
    #              [ 0.1]])
    V = np.array([[ 0.1],
                  [ 0.1],
                  [ 0.1],
                  [ 0.1]])
    #V = np.array([[ 0.1],
    #              [ 0.1],
    #              [ 0.1],
    #              [ 0.1],
    #              [ 0.1]])
    #V = np.array([[ 0.1],
    #              [ 0.1]])
    V = np.kron(V,np.eye(2))
    #
    V2E = np.zeros((2,2,2,2))
    V2E[0,0,1,1] = U
    V2E[1,1,0,0] = U

    no = h1.shape[0] + eb.shape[0]
    FH_list = build_fermion_op(no)
    FH_list_dense = [np.array(FH.todense(),dtype=complex) for FH in FH_list]

    dm, evals, evecs = solve_Hemb_thermal(T, h1, V, eb, V2E, FH_list, verbose=0)

    print('dm=')
    print(dm.real)

    #oms = np.linspace(-2,2,100)
    #eta = 0.1
    ##Gw = compute_Gw_thermal(T, oms, eta, evals, evecs, FH_list_dense)
    #Gw = compute_Gw(oms, eta, evals, evecs, FH_list_dense)
    #plt.plot(oms, Gw.real, label='real')
    #plt.plot(oms, Gw.imag, label='imag')
    #plt.legend()
    #plt.show()

    NomF = 50
    #T = 0.01
    omFs = (2*np.arange(NomF)+1)*np.pi*T
    GomF = compute_GomF(omFs, evals, evecs, FH_list_dense, len(gs_idx))
#
#    plt.plot(omFs, GomF.real, 'bv', label='real')
#    plt.plot(omFs, GomF.imag, 'ro', label='imag')
#    plt.plot(omFs, G0_aim.real, 'b--', label='real')
#    plt.plot(omFs, G0_aim.imag, 'r--', label='imag')
#    plt.legend()
#    plt.show()

