import sklearn.metrics
import sklearn.neighbors
# import matplotlib.pyplot as plt
import scipy.sparse
import scipy.sparse.linalg
import scipy.spatial.distance
import numpy as np
import torch
import math;
from sklearn.linear_model import LinearRegression
from numpy import linalg as LA

def grid(m, dtype=np.float32):
    """Return the embedding of a grid graph."""
    M = m**2
    x = np.linspace(0, 1, m, dtype=dtype)
    y = np.linspace(0, 1, m, dtype=dtype)
    xx, yy = np.meshgrid(x, y)
    z = np.empty((M, 2), dtype)
    z[:, 0] = xx.reshape(M)
    z[:, 1] = yy.reshape(M)
    return z


def distance_scipy_spatial(z, k=4, metric='euclidean', keep_1=False):
    """Compute exact pairwise distances."""
    d = scipy.spatial.distance.pdist(z, metric)
    d = scipy.spatial.distance.squareform(d)
    # k-NN graph.
    if (keep_1):
        start = 0;
    else:
        start = 1;
    idx = np.argsort(d)[:, start:k+1]
    d.sort()
    d = d[:, start:k+1]
    return d, idx


def distance_sklearn_metrics(z, k=4, metric='euclidean'):
    """Compute exact pairwise distances."""
    # d = sklearn.metrics.pairwise.pairwise_distances(
    #         z, metric=metric, n_jobs=1)
    d = z
    # k-NN graph.
    idx = np.argsort(d)[:, 1:k+1]
    d.sort()
    d = d[:, 1:k+1]
    return d, idx


def distance_lshforest(z, k=4, metric='cosine'):
    """Return an approximation of the k-nearest cosine distances."""
    assert metric is 'cosine'
    lshf = sklearn.neighbors.LSHForest()
    lshf.fit(z)
    dist, idx = lshf.kneighbors(z, n_neighbors=k+1)
    assert dist.min() < 1e-10
    dist[dist < 0] = 0
    return dist, idx

# TODO: other ANNs s.a. NMSLIB, EFANNA, FLANN, Annoy, sklearn neighbors, PANN


def adjacency(dist, idx, sigma=None):
    """Return the adjacency matrix of a kNN graph."""
    M, k = dist.shape
    assert M, k == idx.shape
    assert dist.min() >= 0

    # Weights.
    if sigma is None:
        sigma = np.mean(dist[:, -1])**2
    dist = np.exp(- dist**2 / sigma)

    # Weight matrix.
    I = np.arange(0, M).repeat(k)
    J = idx.reshape(M*k)
    V = dist.reshape(M*k)
    W = scipy.sparse.coo_matrix((V, (I, J)), shape=(M, M))

    # No self-connections.
    W.setdiag(0)

    # Non-directed graph.
    bigger = W.T > W
    W = W - W.multiply(bigger) + W.T.multiply(bigger)

    assert W.nnz % 2 == 0
    assert np.abs(W - W.T).mean() < 1e-10
    assert type(W) is scipy.sparse.csr.csr_matrix
    return W

def sign(x):
    if (x >= 0): 
        return 1
    else:
        return -1;

def adjacency_gcn(dist, idx, sigma = None):
    M, k = idx.shape;
    if sigma is None:
        sigma = np.mean(dist[:, -1])**2
    #print('sigma:', sigma);
    L = torch.zeros(M, M)
    for i in range(M):
        for j in range(k):
            t = idx[i][j]
            #randnum = np.random.normal(2.0,0.6)
            L[t][i] = math.exp( -dist[i][j] ** 2/ sigma)
            L[i][t] = math.exp( -dist[i][j] ** 2/ sigma)
    L += torch.eye(M)
    D = L.sum(1).pow(-0.5).view(M);
    D[D == float('Inf')] = 1.
    D = torch.diag(D);
    #L = torch.eye(M)-D.mm(L).mm(D)
    L = D.mm(L).mm(D)
    return L;

def adjacency_cheby(dist, idx, sigma = None):
    M, k = idx.shape;
    if sigma is None:
        sigma = np.mean(dist[:, -1])**2
    #print('sigma:', sigma);
    L = torch.zeros(M, M)
    for i in range(M):
        for j in range(k):
            t = idx[i][j]
            #randnum = np.random.normal(2.0,0.6)
            L[t][i] = math.exp( -dist[i][j] ** 2/ sigma)
            L[i][t] = math.exp( -dist[i][j] ** 2/ sigma)
    D = L.sum(1).pow(-0.5).view(M);
    D[D == float('Inf')] = 1.
    D = torch.diag(D)
    L = torch.eye(M) - 2*D.mm(L).mm(D)
    return L;

def adjacency_polar(dist, idx, theta = 0.0):
    M, k = idx.shape
    L = torch.zeros(M,M)
    for i in range(M):
        for j in range(k):
            t = idx[i][j]
            x = float((t-i)/M)
            y = float((t-i)%M)
            #randnum = np.random.normal(2.0,0.6)
            L[t][i] = np.arctan(x/y)-theta
            L[i][t] = np.pi-L[t][i]
    return L




def adjacency_randomwalk(dist, idx, locations, sigma = None):
    M, k = idx.shape;
    if sigma is None:
        sigma = np.mean(dist[:, -1])**2
    print('sigma:', sigma);
    L = torch.zeros(M, M);
    
    for i in range(M):
        for j in range(k):
            t = idx[i][j]
            xdist = locations[t][0] - locations[i][0]
            ydist = locations[t][1] - locations[i][1]
            dist = xdist ** 2 + ydist ** 2
            L[t][i] = math.exp( -dist/ sigma);
            
    for i in range(M):
        L[i] = L[i] / L[i].sum();
                         
    return L;


# def adjacency_2d(dist, idx, locations, embed_dim):
#     M, k = idx.shape;
#     L = torch.zeros(M, k, embed_dim);
#     max_0 = np.max(np.abs(locations[:,0]));
#     max_1 = np.max(np.abs(locations[:,1]))
#     L_idx = torch.zeros(M, k).long();
    
#     for i in range(M):
#         for j in range(k):
#             t = idx[i][j]
#             L[i][j] = torch.from_numpy(locations[t] - locations[i]);
#             #L[i][j][3] = locations[i][0]; L[i][j][4] = locations[i][1]/max_1;
#             L_idx[i][j] = i * M + t;  
#     L_idx = L_idx.view(M * k);
#     return L_idx, L;

def adjacency_2d(dist, idx, nn, embed_dim=2):
    M, k = idx.shape;
    L = torch.zeros(M, k, embed_dim);
    L_idx = torch.zeros(M, k).long()
    
    for i in range(M):
        for j in range(k):
            t = float(idx[i][j])
            L[i][j][0] = 1/np.sqrt(nn)
            L[i][j][1] = 1/np.sqrt(nn)
            #L[i][j][3] = locations[i][0]; L[i][j][4] = locations[i][1]/max_1;
            L_idx[i][j] = i * M + t;  
    L_idx = L_idx.view(M * k);
    return L_idx, L;




_INF = float('inf')

def adjacency_sigma(dist, idx, locations, sigma = None):
    M, k = idx.shape;
    if sigma is None:
        sigma = np.mean(dist[:, -1])**2
    print('sigma:', sigma);
    L = torch.zeros(M, M);
    
    for i in range(M):
        for j in range(k):
            t = idx[i][j]
            xdist = locations[t][0] - locations[i][0]
            ydist = locations[t][1] - locations[i][1]
            dist = xdist ** 2 + ydist ** 2
            L[t][i] = -dist;
    
    L.masked_fill_(L.eq(0), -_INF);
    return sigma, L;
    
def replace_random_edges(A, noise_level):
    """Replace randomly chosen edges by random edges."""
    M, M = A.shape
    n = int(noise_level * A.nnz // 2)

    indices = np.random.permutation(A.nnz//2)[:n]
    rows = np.random.randint(0, M, n)
    cols = np.random.randint(0, M, n)
    vals = np.random.uniform(0, 1, n)
    assert len(indices) == len(rows) == len(cols) == len(vals)

    A_coo = scipy.sparse.triu(A, format='coo')
    assert A_coo.nnz == A.nnz // 2
    assert A_coo.nnz >= n
    A = A.tolil()

    for idx, row, col, val in zip(indices, rows, cols, vals):
        old_row = A_coo.row[idx]
        old_col = A_coo.col[idx]

        A[old_row, old_col] = 0
        A[old_col, old_row] = 0
        A[row, col] = 1
        A[col, row] = 1

    A.setdiag(0)
    A = A.tocsr()
    A.eliminate_zeros()
    return A


def laplacian(W, normalized=True):
    """Return the Laplacian of the weigth matrix."""

    # Degree matrix.
    d = W.sum(axis=0)

    # Laplacian matrix.
    if not normalized:
        D = scipy.sparse.diags(d.A.squeeze(), 0)
        L = D - W
    else:
        d += np.spacing(np.array(0, W.dtype))
        d = 1 / np.sqrt(d)
        D = scipy.sparse.diags(d.A.squeeze(), 0)
        I = scipy.sparse.identity(d.size, dtype=W.dtype)
        L = I - D * W * D

    # assert np.abs(L - L.T).mean() < 1e-9
    assert type(L) is scipy.sparse.csr.csr_matrix
    return L


def lmax(L, normalized=True):
    """Upper-bound on the spectrum."""
    if normalized:
        return 2
    else:
        return scipy.sparse.linalg.eigsh(
                L, k=1, which='LM', return_eigenvectors=False)[0]


def fourier(L, algo='eigh', k=1):
    """Return the Fourier basis, i.e. the SVD of the Laplacian."""

    def sort(lamb, U):
        idx = lamb.argsort()
        return lamb[idx], U[:, idx]

    if algo is 'eig':
        lamb, U = np.linalg.eig(L.toarray())
        lamb, U = sort(lamb, U)
    elif algo is 'eigh':
        lamb, U = np.linalg.eigh(L.toarray())
    elif algo is 'eigs':
        lamb, U = scipy.sparse.linalg.eigs(L, k=k, which='SM')
        lamb, U = sort(lamb, U)
    elif algo is 'eigsh':
        lamb, U = scipy.sparse.linalg.eigsh(L, k=k, which='SM')

    return lamb, U

def plot_spectrum(L, algo='eig'):
    """Plot the spectrum of a list of multi-scale Laplacians L."""
    # Algo is eig to be sure to get all eigenvalues.
    plt.figure(figsize=(17, 5))
    for i, lap in enumerate(L):
        lamb, U = fourier(lap, algo)
        step = 2**i
        x = range(step//2, L[0].shape[0], step)
        lb = 'L_{} spectrum in [{:1.2e}, {:1.2e}]'.format(i, lamb[0], lamb[-1])
        plt.plot(x, lamb, '.', label=lb)
    plt.legend(loc='best')
    plt.xlim(0, L[0].shape[0])
    plt.ylim(ymin=0)


def lanczos(L, X, K):
    """
    Given the graph Laplacian and a data matrix, return a data matrix which can
    be multiplied by the filter coefficients to filter X using the Lanczos
    polynomial approximation.
    """
    M, N = X.shape
    assert L.dtype == X.dtype

    def basis(L, X, K):
        """
        Lanczos algorithm which computes the orthogonal matrix V and the
        tri-diagonal matrix H.
        """
        a = np.empty((K, N), L.dtype)
        b = np.zeros((K, N), L.dtype)
        V = np.empty((K, M, N), L.dtype)
        V[0, ...] = X / np.linalg.norm(X, axis=0)
        for k in range(K-1):
            W = L.dot(V[k, ...])
            a[k, :] = np.sum(W * V[k, ...], axis=0)
            W = W - a[k, :] * V[k, ...] - (
                    b[k, :] * V[k-1, ...] if k > 0 else 0)
            b[k+1, :] = np.linalg.norm(W, axis=0)
            V[k+1, ...] = W / b[k+1, :]
        a[K-1, :] = np.sum(L.dot(V[K-1, ...]) * V[K-1, ...], axis=0)
        return V, a, b

    def diag_H(a, b, K):
        """Diagonalize the tri-diagonal H matrix."""
        H = np.zeros((K*K, N), a.dtype)
        H[:K**2:K+1, :] = a
        H[1:(K-1)*K:K+1, :] = b[1:, :]
        H.shape = (K, K, N)
        Q = np.linalg.eigh(H.T, UPLO='L')[1]
        Q = np.swapaxes(Q, 1, 2).T
        return Q

    V, a, b = basis(L, X, K)
    Q = diag_H(a, b, K)
    Xt = np.empty((K, M, N), L.dtype)
    for n in range(N):
        Xt[..., n] = Q[..., n].T.dot(V[..., n])
    Xt *= Q[0, :, np.newaxis, :]
    Xt *= np.linalg.norm(X, axis=0)
    return Xt  # Q[0, ...]


def rescale_L(L, lmax=2):
    """Rescale the Laplacian eigenvalues in [-1,1]."""
    M, M = L.shape
    I = scipy.sparse.identity(M, format='csr', dtype=L.dtype)
    L /= lmax / 2
    L -= I
    return L


def chebyshev(L, X, K):
    """Return T_k X where T_k are the Chebyshev polynomials of order up to K.
    Complexity is O(KMN)."""
    M, N = X.shape
    assert L.dtype == X.dtype

    # L = rescale_L(L, lmax)
    # Xt = T @ X: MxM @ MxN.
    Xt = np.empty((K, M, N), L.dtype)
    # Xt_0 = T_0 X = I X = X.
    Xt[0, ...] = X
    # Xt_1 = T_1 X = L X.
    if K > 1:
        Xt[1, ...] = L.dot(X)
    # Xt_k = 2 L Xt_k-1 - Xt_k-2.
    for k in range(2, K):
        Xt[k, ...] = 2 * L.dot(Xt[k-1, ...]) - Xt[k-2, ...]
    return Xt


def picfast_a(W, v0, default_conv, maxit):
    n = W.shape[0]
    vt = v0
    dt = np.ones(n)
    dtp = np.zeros(n)
    i = 0;
    while(np.amax(np.absolute(dt-dtp))>default_conv and i < maxit):
        vtp = vt
        dtp = dt
        vt = W.dot(vt)
        vt = vt/np.sum(vt)
        dt = np.absolute(vt-vtp)
        i += 1
    return vt, i



def dpie_a(W, default_conv, maxit, nd, max_pie, eta):
    nrow = W.shape[0]
    vRD = np.random.rand(nrow,nd)
    v0 = np.random.rand(nrow)
    DPIE_old =  np.zeros((nrow, nd))
    DPIE_old_it = np.zeros(nd)
    BasicPIE, num_it = picfast_a(W, v0, default_conv, maxit)
    DPIE_old[:,0] = BasicPIE
    DPIE_old_it[0] = num_it
    for i in range(1,nd):
        num_it = 0
        mindt = np.zeros(nrow)
        vt = vRD[:,i-1]
        critie1 = 10000
        while((critie1>(default_conv*i)) and (maxit>num_it)):
            vtp = vt
            mindtp =  mindt
            vt = W.dot(vt) 
            vt = vt/np.sum(np.absolute(vt))
            mindt = np.absolute(vt - vtp)
            num_it += 1
            critie1 = np.amax(np.absolute(mindt - mindtp))
        DPIE_old[:,i] = vt
        DPIE_old_it[i] = num_it
    DPIE = BasicPIE.reshape(-1,1)
    num_pie = 1
    #print('IT',DPIE_old_it)
    for i in range(1,nd):
        reg = LinearRegression(fit_intercept = False)
        reg.fit(DPIE,DPIE_old[:,i])
        tmpV = DPIE_old[:,i] - reg.predict(DPIE)
        reg_err = np.sum(np.absolute(tmpV))/np.sum(DPIE_old[:,i])
        if (reg_err>=eta):
            tmpV = tmpV/np.sum(np.absolute(tmpV))
            DPIE = np.concatenate((DPIE, tmpV.reshape(-1,1)), axis=1)
            #DPIE_it = np.concatenate((DPIE_it,, axis=1))
            #print(i, DPIE_old_it[i])
            num_pie += 1
            if num_pie >= max_pie:
                break
    #DPIV and Orthogonalization
    print('DPIE', DPIE.shape)
    lamb = np.empty(DPIE.shape[1])
    for i in range(DPIE.shape[1]):
        #lamb[i] = DRev.dot(NormF.dot(Data.dot(Data.T.dot(NormF.dot(DPIE[:,i]))))) 
        lamb[i] = np.matmul(DPIE[:,i],W.dot(DPIE[:,i]))/np.matmul(DPIE[:,i],DPIE[:,i])
        #print(lamb[i])
    #Orthogonalization
    P = np.matmul(DPIE.T, DPIE)
    Pw, Pv = np.linalg.eigh(P)
    vw = Pv*np.sqrt(Pw)
    B = np.matmul(vw.T*lamb, vw)
    Bw, Bv = np.linalg.eigh(B)
    DPIEorth = np.matmul(DPIE,np.matmul(Pv/np.sqrt(Pw), Bv))
    return DPIE, DPIEorth, lamb


def picfast(Data, NormF, DRev, DRevsqrt, v0, default_conv, maxit,normalized):
    n = Data.shape[0]
    vt = v0
    dt = np.ones(n)
    dtp = np.zeros(n)
    i = 0;
    while(np.amax(np.absolute(dt-dtp))>default_conv and i < maxit):
        vtp = vt
        dtp = dt
        #vt = DRev.dot(NormF.dot(Data.dot(Data.T.dot(NormF.dot(vt)))))
        if normalized == 'rw':
            vt = DRev*NormF*(Data.dot(Data.T.dot(NormF*vt)))
        else:
            vt = DRevsqrt*NormF*(Data.dot(Data.T.dot(DRevsqrt*NormF*vt)))
        vt = vt/np.sum(vt)
        dt = np.absolute(vt-vtp)
        i += 1
    return vt, i

def prep_cos(Data):
    ep = 1e-16
    n = Data.shape[0]
    if scipy.sparse.issparse(Data):
        NormF = 1/(scipy.sparse.linalg.norm(Data, axis=1)+ep)
    else:
        NormF = 1/(LA.norm(Data, axis=1)+ep)
    D = NormF*(Data.dot(Data.T.dot(NormF)))
    Dsqrt = np.sqrt(D)
    DRev = 1/(D+ep)
    DRevsqrt = 1/(Dsqrt+ep)
    return DRev, NormF, DRevsqrt



def MatrixFastAppr(X, sigma, L):
    nrow, ncol = X.shape
    W = np.fft.fft((np.random.normal(0, 1, (L,ncol))/sigma).T, axis=0).T.real
    b = 2*np.pi*np.random.rand(L,1)
    b = np.tile(b,(1,nrow))
    X_prj = (math.sqrt(2/float(L))*np.cos((X.dot(W.T)).T+b)).T;
    DRev, NormF, D = prep_cos(X_prj)
    return DRev, NormF, D, X_prj


def dpie(Data, NormF, DRev, DRevsqrt, default_conv, maxit, nd, max_pie, eta, normalized):
    nrow = Data.shape[0]
    vRD = np.random.rand(nrow,nd)
    v0 = np.random.rand(nrow)
    DPIE_old =  np.zeros((nrow, nd))
    DPIE_old_it = np.zeros(nd)
    BasicPIE, num_it = picfast(Data, NormF, DRev, DRevsqrt, v0, default_conv, maxit, normalized)
    DPIE_old[:,0] = BasicPIE
    DPIE_old_it[0] = num_it
    for i in range(1,nd):
        num_it = 0
        mindt = np.zeros(nrow)
        vt = vRD[:,i-1]
        critie1 = 10000
        while((critie1>(default_conv*i)) and (maxit>num_it)):
            vtp = vt
            mindtp =  mindt
            #vt = DRev.dot(NormF.dot(Data.dot(Data.T.dot(NormF.dot(vt))))) 
            #vt = DRev*(NormF*(Data.dot(Data.T.dot(NormF*vt)))-vt)
            if normalized == 'rw':
                vt = DRev*NormF*(Data.dot(Data.T.dot(NormF*vt)))
            else:
                vt = DRevsqrt*NormF*(Data.dot(Data.T.dot(DRevsqrt*NormF*vt)))
            vt = vt/np.sum(np.absolute(vt))
            mindt = np.absolute(vt - vtp)
            num_it += 1
            critie1 = np.amax(np.absolute(mindt - mindtp))
        DPIE_old[:,i] = vt
        DPIE_old_it[i] = num_it
    DPIE = BasicPIE.reshape(-1,1)
    num_pie = 1
    print('IT',DPIE_old_it)
    for i in range(1,nd):
        reg = LinearRegression(fit_intercept = False)
        #print(DPIE.shape, DPIE_old[:,i].shape)
        reg.fit(DPIE,DPIE_old[:,i])
        tmpV = DPIE_old[:,i] - reg.predict(DPIE)
        reg_err = np.sum(np.absolute(tmpV))/np.sum(DPIE_old[:,i])
        #print(reg_err)
        if (reg_err>=eta):
            tmpV = tmpV/np.sum(np.absolute(tmpV))
            DPIE = np.concatenate((DPIE, tmpV.reshape(-1,1)), axis=1)
            #DPIE_it = np.concatenate((DPIE_it,, axis=1))
            #print(i, DPIE_old_it[i])
            num_pie += 1
            if num_pie >= max_pie:
                break
    #DPIV and Orthogonalization
    print('DPIE', DPIE.shape)
    lamb = np.empty(DPIE.shape[1])
    for i in range(DPIE.shape[1]):
        #lamb[i] = DRev.dot(NormF.dot(Data.dot(Data.T.dot(NormF.dot(DPIE[:,i]))))) 
        lamb[i] = np.matmul(DPIE[:,i],DRev*(NormF*(Data.dot(Data.T.dot(NormF*DPIE[:,i])))))/\
                  np.matmul(DPIE[:,i],DPIE[:,i])
        print(lamb[i])
    #Orthogonalization
    P = np.matmul(DPIE.T, DPIE)
    Pw, Pv = np.linalg.eigh(P)
    vw = Pv*np.sqrt(Pw)
    B = np.matmul(vw.T*lamb, vw)
    Bw, Bv = np.linalg.eigh(B)
    DPIEorth = np.matmul(DPIE,np.matmul(Pv/np.sqrt(Pw), Bv))
    return DPIE, DPIEorth, lamb
