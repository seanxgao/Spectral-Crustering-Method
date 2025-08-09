import numpy as np
from scipy.sparse import linalg as LA
from scipy.sparse import csr_matrix
import time
import matplotlib.pyplot as plt
import random

def bicut_group(L):
    """
    Enhanced spectral clustering function that returns both sign-based and optimal cuts.
    
    Args:
        L: numpy array, the (sub-)graph Laplacian matrix
    
    Returns:
        tuple: (first_group, second_group) where second_group may be empty
    """
    
    n = L.shape[0]

    # Basis steps
    if n == 0:
        raise ValueError("The Laplacian matrix is empty.") 
    if n == 1:
        return [0], []
    if n == 2:
        return [0], [1]
    
    # Get Fiedler vector and sort vertices
    _, eigenvecs = LA.eigsh(L,k=2,which = 'SA')
    
    # Find the second smallest eigenvalue (Fiedler value)
    fiedler_vector = eigenvecs[:, 1]
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
        """
        Extract the order of singleton nodes (leaves with single elements)
        in the order they appear in a tree traversal.
        
        Returns:
            list: Order of singleton vertices as they appear in the tree
        """
        order = []
        
        def collect_singletons(node):
            # If this is a leaf with exactly one element, add it to order
            if node.is_leaf():
                order.extend(node.indices)
            else:
                # Traverse children in order (left first, then right)
                if node.left:
                    collect_singletons(node.left)
                if node.right:
                    collect_singletons(node.right)
        
        collect_singletons(self)
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

def treebuilder(laplacian_matrix, thre = None, indices=None):
    """
    Recursively apply bi-cut to create a tree structure.
    
    Args:
        laplacian_matrix: numpy array, the full graph Laplacian matrix
        indices: list of indices to process (None means all vertices)
    
    Returns:
        BiCutNode: Root of the bi-cut tree
    """
    # Initialize indices if not provided
    if indices is None:
        indices = list(range(laplacian_matrix.shape[0]))

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
    
    # Apply bi-cut on submatrix
    first_group_local, second_group_local = bicut_group(laplacian_matrix)
    
    # Convert local indices back to global indices
    first_group = [indices[i] for i in first_group_local]
    second_group = [indices[i] for i in second_group_local]
    
    # If second group is empty, this is a leaf node
    if not second_group:
        return BiCutNode(indices)
    
    # Create current node
    node = BiCutNode(indices)
    
    # Recursively process subgroups
    node.left = treebuilder(laplacian_matrix[np.ix_(first_group_local, first_group_local)], thre, first_group)
    node.right = treebuilder(laplacian_matrix[np.ix_(second_group_local, second_group_local)], thre, second_group)
    
    return node




