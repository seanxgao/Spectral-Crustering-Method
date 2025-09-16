import numpy as np
from scipy.sparse import linalg as LA
from scipy.sparse import csr_matrix
import time
import matplotlib.pyplot as plt
import random
import sys
from scipy import sparse as sp
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh  # sparse solver

def bicut_group(L):
    """
    Enhanced spectral clustering function that returns both sign-based and optimal cuts.
    
    Args:
        L: numpy array, the (sub-)graph Laplacian matrix
    
    Returns:
        tuple: (first_group, second_group) where second_group may be empty
    """
    # if structure not in ("dense", "sparse"):
    #     raise ValueError("structure must be 'dense' or 'sparse'.")

    n = L.shape[0]

    # Basis steps
    if n == 0:
        raise ValueError("The Laplacian matrix is empty.") 
    if n == 1:
        return [0], []
    if n == 2:
        return [0], [1]
    
    _, vecs = eigsh(L, k=2, which='SA')
    # Find the second smallest eigenvalue (Fiedler value)
    fiedler_vector = vecs[:, 1]
    sorted_args = np.argsort(fiedler_vector)
    
    # Reorder adjacency matrix
    adj = -L[np.ix_(sorted_args,sorted_args)]  # Full adjacency matrix
    
    # Find best cut
    ind = np.arange(1, n)
    upper_tri_sums = np.array([np.sum(adj[i:, :i]) for i in ind])
    qualities = upper_tri_sums / (ind * (n - ind))
    
    best_cut = np.argmin(qualities) + 1
    
    # Get the groups based on sorted indices
    first_group = sorted_args[:best_cut]
    second_group = sorted_args[best_cut:]

    # Continue with the split
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

def treebuilder(L, thre = None, indices=None, structure = "sparse"):
    """
    Recursively apply bi-cut to create a tree structure.
    
    Args:
        L: numpy array, the full graph Laplacian matrix
        indices: list of indices to process (None means all vertices)
    
    Returns:
        BiCutNode: Root of the bi-cut tree
    """
    if structure not in ("dense", "sparse"):
        raise ValueError("structure must be 'dense' or 'sparse'.")

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

    # if structure == "dense":
    #     # Convert to dense (if sparse)
    #     L = L.toarray() if sp.issparse(L) else L
    # else:
    #     L = L if sp.issparse(L) else csr_matrix(L)
    #     # L = L.astype(np.float32)
    
    # Apply bi-cut on submatrix
    first_group_local, second_group_local = bicut_group(L)
    
    # Convert local indices back to global indices
    first_group = [indices[i] for i in first_group_local]
    second_group = [indices[i] for i in second_group_local]
    
    # If second group is empty, this is a leaf node
    if not second_group:
        return BiCutNode(indices)
    
    # Create current node
    node = BiCutNode(indices)

    leftmatrix  = L[np.ix_(first_group_local,  first_group_local)]
    rightmatrix = L[np.ix_(second_group_local, second_group_local)]
    
    if structure == "sparse":
        degdiff = np.asarray(leftmatrix.sum(axis=1)).ravel()
        leftmatrix.setdiag(leftmatrix.diagonal() - degdiff)
        leftmatrix.eliminate_zeros()
    
        degdiff = np.asarray(rightmatrix.sum(axis=1)).ravel()
        rightmatrix.setdiag(rightmatrix.diagonal() - degdiff)
        rightmatrix.eliminate_zeros()
    # else:
        # degdiff = leftmatrix.sum(axis=1)
        # leftmatrix = leftmatrix - np.diag(degdiff)
    
        # degdiff = rightmatrix.sum(axis=1)
        # rightmatrix = rightmatrix - np.diag(degdiff)
        
    # Recursively process subgroups
    node.left = treebuilder(leftmatrix, thre, first_group, structure)
    node.right = treebuilder(rightmatrix, thre, second_group, structure)
    
    return node

import os, math
from concurrent.futures import ThreadPoolExecutor

def treebuilder_parallel(L, thre=None, workers=None, max_parallel_depth=None):
    """
    简洁并行版 BiCut 构树（线程池）
    参数：
      - L: 整体拉普拉斯矩阵（numpy 或 scipy.sparse CSR/CSC）
      - thre: 叶子阈值（子块规模 ≤ thre 就不再切）
      - workers: 并行线程数（默认=CPU核数）
      - max_parallel_depth: 最大并行深度，超过就转串行，防止任务过多
    返回：
      - BiCutNode 根节点
    """
    if workers is None:
        workers = max(os.cpu_count() or 1, 1)

    # 经验：2^d >= workers ⇒ d ≈ ceil(log2(workers))
    if max_parallel_depth is None:
        max_parallel_depth = max(1, int(math.ceil(math.log2(workers))))

    # 内部递归函数：当 depth < max_parallel_depth 时并行左右子树，否则串行
    def _build(L_sub, indices, depth, executor):
        n = len(indices)
        # —— 基本情形：小块或阈值内就收叶子
        if n == 0:
            raise ValueError("The matrix is empty.")
        if n == 1:
            return BiCutNode(indices)
        if n == 2:
            return BiCutNode(indices, BiCutNode([indices[0]]), BiCutNode([indices[1]]))
        if (thre is not None) and (n <= thre):
            return BiCutNode(indices)

        # —— 试着再切一刀
        g1_local, g2_local = bicut_group(L_sub)
        if len(g2_local) == 0: 
            return BiCutNode(indices)

        # 局部→全局 索引
        g1 = [indices[i] for i in g1_local]
        g2 = [indices[i] for i in g2_local]

        # 子块拉普拉斯（与局部索引对齐）
        L11 = L_sub[np.ix_(g1_local, g1_local)]
        L22 = L_sub[np.ix_(g2_local, g2_local)]

        # —— 到这一步需要构左右子树：并行或串行
        if depth < max_parallel_depth and workers > 1 and executor is not None:
            # 并行提交左右子树
            f_left  = executor.submit(_build, L11, g1, depth + 1, executor)
            f_right = executor.submit(_build, L22, g2, depth + 1, executor)
            left  = f_left.result()
            right = f_right.result()
        else:
            # 超过并行深度，直接串行，避免任务过多
            left  = _build(L11, g1, depth + 1, executor)
            right = _build(L22, g2, depth + 1, executor)

        return BiCutNode(indices, left, right)

    # 线程池外层只开一次；把根任务丢进去递归构建
    with ThreadPoolExecutor(max_workers=workers) as ex:
        root = _build(L, list(range(L.shape[0])), 0, ex)
    return root




