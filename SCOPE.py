import numpy as np
from scipy.sparse import csr_matrix
from scipy import sparse as sp
from scipy.sparse.linalg import eigsh as cpu_eigsh
import os, math
from concurrent.futures import ThreadPoolExecutor
import cupy as cp
import cupyx.scipy.sparse as csp
from cupyx.scipy.sparse.linalg import eigsh as gpu_eigsh


def best_cut_finder(adj, gpu: bool, sparse: bool) -> int:
    """
    Compute the best cut index for symmetric adjacency matrices.

    Args:
        adj: matrix (dense or CSR; NumPy/SciPy on CPU or CuPy/cupyx on GPU)
        gpu:  use GPU (CuPy) kernels if True, else CPU (NumPy/SciPy)
        sparse: treat/convert to CSR and use O(nnz) path if True; else dense O(n^2)

    Returns:
        int: best cut index in [1, n-1]
    """
    # ---- helpers ----
    def _best_cut_dense_np(A):
        A = A.toarray() if hasattr(A, "toarray") else np.asarray(A)
        n = A.shape[0]
        if n < 2:
            return 1
        ps = np.zeros((n+1, n+1), dtype=np.float64)
        ps[1:, 1:] = A
        ps = ps.cumsum(0).cumsum(1)
        i = np.arange(1, n, dtype=np.int64)
        q = (ps[n, i] - ps[i, i]) / (i * (n - i))
        return int(np.argmin(q) + 1)

    def _best_cut_dense_cp(A):
        A = A if isinstance(A, cp.ndarray) else cp.asarray(A)
        n = A.shape[0]
        if n < 2:
            return 1
        ps = cp.zeros((n+1, n+1), dtype=cp.float64)  # accumulate in fp64
        ps[1:, 1:] = A
        ps = ps.cumsum(0).cumsum(1)
        i = cp.arange(1, n, dtype=cp.int64)
        q = (ps[n, i] - ps[i, i]) / (i * (n - i))
        return int(cp.asnumpy(cp.argmin(q)) + 1)

    def _best_cut_sparse_sp(M):
        # ensure CPU CSR
        csr = M.tocsr() if sp.issparse(M) else sp.csr_matrix(M)
        n = csr.shape[0]
        if n < 2:
            return 1
        coo = csr.tocoo(copy=False)
        r, c = coo.row, coo.col
        v = coo.data.astype(np.float64, copy=False)
        mask = c < r  # strict lower triangle (symmetric input)
        if mask.any():
            start = (c[mask] + 1).astype(np.int64, copy=False)
            end   = (r[mask] + 1).astype(np.int64, copy=False)
            w     = v[mask]
            diff = np.zeros(n + 1, dtype=np.float64)
            np.add.at(diff, start,  w)
            np.add.at(diff, end,   -w)
            pref = np.cumsum(diff)[:-1]
        else:
            pref = np.zeros(n, dtype=np.float64)
        i = np.arange(1, n, dtype=np.int64)
        q = pref[i] / (i * (n - i))
        return int(np.argmin(q) + 1)

    def _best_cut_sparse_csp(M):
        # ensure GPU CSR
        if isinstance(M, csp.csr_matrix):
            csr = M
        elif sp.issparse(M):
            cpu = M.tocsr()
            csr = csp.csr_matrix((cp.asarray(cpu.data, dtype=cp.float32),
                                  cp.asarray(cpu.indices),
                                  cp.asarray(cpu.indptr)),
                                 shape=cpu.shape)
        elif isinstance(M, cp.ndarray):
            csr = csp.csr_matrix(M)
        else:
            csr = csp.csr_matrix(cp.asarray(M))
        n = csr.shape[0]
        if n < 2:
            return 1
        coo = csr.tocoo(copy=False)
        r, c, v = coo.row, coo.col, coo.data
        mask = c < r
        if mask.any():
            start = (c[mask] + 1).astype(cp.int64, copy=False)
            end   = (r[mask] + 1).astype(cp.int64, copy=False)
            w     = v[mask].astype(cp.float64, copy=False)   # accumulate in fp64
            diff = cp.zeros(n + 1, dtype=cp.float64)
            cp.add.at(diff, start,  w)
            cp.add.at(diff, end,   -w)
            pref = cp.cumsum(diff)[:-1]
        else:
            pref = cp.zeros(n, dtype=cp.float64)
        i = cp.arange(1, n, dtype=cp.int64)
        q = pref[i] / (i * (n - i))
        return int(cp.asnumpy(cp.argmin(q)) + 1)

    # ---- dispatch ----
    if gpu:
        return _best_cut_sparse_csp(adj) if sparse else _best_cut_dense_cp(adj)
    else:
        return _best_cut_sparse_sp(adj)   if sparse else _best_cut_dense_np(adj)

def eigen_decomposition(L, gpu: bool, sparse: bool, k = 2):
    L_csr = matrixtype(L, gpu=gpu, sparse=sparse)
    zero_tol = 1e-2
    n = L_csr.shape[0]

    from matrix_master import visualize_laplacian_matrix
    def _solve(k_now):
        if gpu:
            # gpu_eigsh assumed SciPy-like signature
            w, v = gpu_eigsh(L_csr, k=k_now, which="SA")
            w = cp.asnumpy(w)
            v = cp.asnumpy(v)
        else:
            w, v = cpu_eigsh(L_csr, k=k_now, which="SA")
        return w, v

    # ---- first try: k (default 2) ----
    from scipy.sparse.linalg import ArpackNoConvergence, ArpackError
    while True:
        try:
            if k > n-1:
                k = n-1
            w, v = _solve(k)
            nz = np.where(w > zero_tol)[0]
            print(nz)
            if nz.size:
                print(nz[0], "th eigenvalue is the first non-zero one, where k =", k, "n =", n)
                return v[:, nz[0]]
            if k == n-1:
                print(matrixtype(L, gpu=False, sparse=False))
                raise ValueError("All eigenvalues are numerically zero; cannot proceed.")
            else:
                k = min(k*2, n - 1)
            if matrixtype(L, gpu=False, sparse=False).diagonal().sum() < 1:
                raise Warning("zero matrix")
        except ArpackNoConvergence:
            print("ARPACK did not converge with k =", k)
            k = min(k*2, n - 1)
class pilot:
    def __init__(self):
        self.parallel = False

        self.sparsethred = 0
        self.gputhred4eig = 0
        self.gputhred4cut = 0

        self.densityhasreached = False

        self.eig = None
        self.cut = None
        self.sparse = None

    def set_spthre(self, thr=0.25):
        def sparse (L):
            if sparse_score(L) <= thr:
                return True
            else:
                self.sparse = lambda L: False
                return False
        self.sparse = sparse
        self.sparsethred = thr

    def set_gputhre(self, thred_eig = 10000, thred_cut=1000):
        if self.parallel:
            self.eig = lambda L: False
            self.cut = lambda L: False
        else:
            def eig (L):
                if L.shape[0] >= thred_eig:
                    return True
                else:
                    self.eig = lambda L: False
                    return False
            self.eig = eig

            def cut (L):
                if L.shape[0] >= thred_cut:
                    return True
                else:
                    self.cut = lambda L: False
                    return False
            self.cut = cut

        self.gputhred4eig = thred_eig
        self.gputhred4cut = thred_cut
    def copy(self):
        import copy
        return copy.copy(self)

def bicut_group(L, gpueigen=False, gpucut=False, sparse=False):
    """
    Enhanced spectral clustering function that returns both sign-based and optimal cuts.
    
    Args:
        L: numpy array, the (sub-)graph Laplacian matrix
    
    Returns:
        tuple: (first_group, second_group) where second_group may be empty
    """
    n = L.shape[0]
    if n == 0:
        raise ValueError("The Laplacian matrix is empty.")
    if n == 1:
        return [0], []
    if n == 2:
        return [0], [1]

    fiedler = eigen_decomposition(L, gpu=gpueigen, sparse=sparse)
    
    sorted_args = np.argsort(fiedler)
    
    adj = -L[np.ix_(sorted_args, sorted_args)]
    
    best_cut = best_cut_finder(adj, gpu=gpucut, sparse=sparse)

    first_group = sorted_args[:best_cut]
    second_group = sorted_args[best_cut:]
    if 0 in first_group:
        return first_group, second_group
    return second_group, first_group


class BiCutNode:
    """Node class for the bi-cut tree structure"""
    def __init__(self, indices, left=None, right=None, parent=None):
        self.indices = indices  # List of vertex indices in this node
        self.left = left       # Left subtree
        self.right = right     # Right subtree

    def is_leaf(self):
        return self.left is None and self.right is None
    
    def get_order(self):
        order = []
        def dfs(node):
            if node.is_leaf():
                order.extend(node.indices)
            else:
                if node.left: dfs(node.left)
                if node.right: dfs(node.right)
        dfs(self)
        return order
    
    def print_fancy_tree(self, prefix="", is_last=True, is_root=True):
        """Print tree with fancy box-drawing characters"""
        if is_root:
            print("┌─ BiCut Tree Structure")
        
        connector = "├─" if is_root else ("└─" if is_last else "├─")
        indices_str = f"[{', '.join(map(str, sorted(self.indices)))}]"
        
        print(f"{prefix}{connector} {indices_str}")
        
        new_prefix = prefix + ("│  " if is_root else ("   " if is_last else "│  "))
        children = [child for child in [self.left, self.right] if child is not None]
        
        for i, child in enumerate(children):
            is_last_child = (i == len(children) - 1)
            child.print_fancy_tree(new_prefix, is_last_child, False)

def treebuilder(L, thre = None, indices=None, parallel = True, manager = None):
    """
    Recursively apply bi-cut to create a tree structure.
    
    Args:
        L: numpy array, the full graph Laplacian matrix
        indices: list of indices to process (None means all vertices)
    
    Returns:
        BiCutNode: Root of the bi-cut tree
    """

    if manager is None:
        manager = pilot()
        manager.parallel = parallel
        manager.set_spthre(0.25)
        manager.set_gputhre(10000, 1000)
    
    if manager.sparse is None or manager.eig is None or manager.cut is None:
        manager.parallel = parallel
        manager.set_spthre(0.25)
        manager.set_gputhre(10000, 1000)

    def _make_sub_laplacian_blocks(L_sub, g1_local, g2_local):
        
        idx1 = np.asarray(g1_local)
        idx2 = np.asarray(g2_local) 
        
        L_sub = matrixtype(L_sub, gpu=False, sparse=False)  # Ensure CPU dense for indexing
        L11 = L_sub[np.ix_(idx1, idx1)].copy()
        L22 = L_sub[np.ix_(idx2, idx2)].copy()

        diff1 = L11.sum(axis=1)
        diff2 = L22.sum(axis=1)
        L11 = L11 - np.diag(diff1)
        L22 = L22 - np.diag(diff2)
        return L11, L22

    # Initialize indices if not provided
    if indices is None:
        indices = list(range(L.shape[0]))

    n= len(indices)

    # Base case: empty or single vertex
    if n == 0:
        raise ValueError("The matrix is empty.")
    if n == 1:
        return BiCutNode(indices)
    if n == 2:
        return BiCutNode(indices,BiCutNode([indices[0]]),BiCutNode([indices[1]]))
    
    if thre != None and n <= thre:
        return BiCutNode(indices)

    if parallel:
        workers = max(os.cpu_count() or 1, 1)
        max_parallel_depth = max(1, int(math.ceil(math.log2(workers))))
        
    # === 递归构建（根据 parallel 选择并/串行）===
    def _build(L_sub, idxs, depth, executor=None,manager=None):
        n = len(idxs)
        if n == 0:
            raise ValueError("The matrix is empty, should not happen.")
        if n == 1:
            return BiCutNode(idxs)
        if n == 2:
            return BiCutNode(idxs, BiCutNode([idxs[0]]), BiCutNode([idxs[1]]))
        if (thre is not None) and (n <= thre):
            return BiCutNode(idxs)

        gpueig = manager.eig(L_sub)
        gpucut = manager.cut(L_sub)
        sparse = manager.sparse(L_sub)
        # bi-cut

        from scipy.sparse.linalg import ArpackError
        try:
            g1_local, g2_local = bicut_group(L_sub, gpueigen=gpueig, gpucut=gpucut, sparse=sparse)
            if not g1_local.shape[0] or not g2_local.shape[0]:
                raise ValueError("Bi-cut resulted in an empty group; cannot proceed.")
        except ArpackError as e:
            print(f"Error occurred during bi-cut: {e}")
            return

        # 局部→全局
        g1 = [idxs[i] for i in g1_local]
        g2 = [idxs[i] for i in g2_local]

        # 子块拉普拉斯
        L11, L22 = _make_sub_laplacian_blocks(L_sub, g1_local, g2_local)

        # 递归：并行 or 串行
        if parallel and (depth < max_parallel_depth) and (executor is not None) and (workers > 1):
            f_left  = executor.submit(_build, L11, g1, depth + 1, executor, manager.copy())
            f_right = executor.submit(_build, L22, g2, depth + 1, executor, manager.copy())
            left  = f_left.result()
            right = f_right.result()
        else:
            left  = _build(L11, g1, depth + 1, executor, manager.copy())
            right = _build(L22, g2, depth + 1, executor, manager.copy())

        return BiCutNode(idxs, left, right)

    # 顶层入口：并行就开线程池；串行就直接跑
    if parallel:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            return _build(L, indices, 0, ex, manager)
    else:
        return _build(L, indices, 0, None, manager)

def matrixtype(L, sparse: bool, gpu: bool):
    """
    将 L 转成以下四种之一（仅由 sparse / gpu 决定）：
      - gpu=True,  sparse=True  -> cupyx.scipy.sparse.csr_matrix
      - gpu=True,  sparse=False -> cupy.ndarray
      - gpu=False, sparse=True  -> scipy.sparse.csr_matrix
      - gpu=False, sparse=False -> numpy.ndarray

    允许的输入类型：
      - cupyx.scipy.sparse.csr_matrix
      - cupy.ndarray
      - numpy.ndarray
      - scipy.sparse.csr_matrix

    约定：sp, cp, csp 已成功导入：
      import scipy.sparse as sp
      import cupy as cp
      import cupyx.scipy.sparse as csp
    同时假定已 import numpy as np
    """
    # ==== 目标：GPU + SPARSE -> cupyx.scipy.sparse.csr_matrix ====
    if gpu and sparse:
        if csp.isspmatrix(L):
            return L.tocsr()                         # 已是 GPU 稀疏 -> CSR
        if sp.issparse(L):
            return csp.csr_matrix(L)                 # CPU 稀疏 -> 复制到 GPU
        if isinstance(L, cp.ndarray):
            return csp.csr_matrix(L)                 # GPU 稠密 -> GPU CSR
        # 其余视为 CPU 稠密 / array-like
        return csp.csr_matrix(cp.asarray(L))         # CPU 稠密 -> GPU CSR

    # ==== 目标：GPU + DENSE -> cupy.ndarray ====
    if gpu and not sparse:
        if isinstance(L, cp.ndarray):
            return L                                  # 已是 GPU 稠密
        if csp.isspmatrix(L):
            return L.toarray()                        # GPU 稀疏 -> GPU 稠密(cupy)
        if sp.issparse(L):
            return cp.asarray(L.toarray())            # CPU 稀疏 -> CPU 稠密 -> GPU 稠密
        # 其余视为 CPU 稠密 / array-like
        return cp.asarray(L)                          # CPU 稠密 -> GPU 稠密

    # ==== 目标：CPU + SPARSE -> scipy.sparse.csr_matrix ====
    if (not gpu) and sparse:
        if sp.issparse(L):
            return sp.csr_matrix(L)                   # 已是 CPU 稀疏 -> CSR
        if csp.isspmatrix(L):
            return L.get().tocsr()                    # GPU 稀疏 -> CPU 稀疏
        if isinstance(L, cp.ndarray):
            return sp.csr_matrix(cp.asnumpy(L))       # GPU 稠密 -> CPU 稀疏
        # 其余视为 CPU 稠密 / array-like
        return sp.csr_matrix(L)                       # CPU 稠密 -> CPU 稀疏

    # ==== 目标：CPU + DENSE -> numpy.ndarray ====
    # gpu=False and sparse=False
    if sp.issparse(L):
        return L.toarray()                            # CPU 稀疏 -> CPU 稠密(numpy)
    if csp.isspmatrix(L):
        return L.get().toarray()                      # GPU 稀疏 -> CPU 稠密(numpy)
    if isinstance(L, cp.ndarray):
        return cp.asnumpy(L)                          # GPU 稠密 -> CPU 稠密
    # 其余视为 CPU 稠密 / array-like
    return np.asarray(L)                               # 保证是 numpy.ndarray

def sparse_score(L):
    """
    Note:
        only for laplacian matrix
    """
    n = L.shape[0]
    if sp.issparse(L):  # CPU sparse
        s = float(L.diagonal().sum(dtype=np.float64))
    elif csp.isspmatrix(L):  # GPU sparse
        s = float(cp.asnumpy(L.diagonal().sum(dtype=cp.float64)))
    elif isinstance(L, cp.ndarray):  # GPU dense
        s = float(cp.asnumpy(L.diagonal().sum(dtype=cp.float64)))
    else:  # CPU dense / array-like
        X = np.asarray(L)
        s = float(X.diagonal().sum(dtype=np.float64))

    return s / (n * n)