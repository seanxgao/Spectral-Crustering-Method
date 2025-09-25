from SCOPE import *
import matplotlib.pyplot as plt
import random

def generate_test_laplacian(size1, size2, prob1, prob2, prob_between):
    """
    Generate a random Laplacian matrix for testing spectral clustering.
    
    Args:
        size1: size of first group
        size2: size of second group  
        prob1: probability of edge within first group
        prob2: probability of edge within second group
        prob_between: probability of edge between groups
    
    Returns:
        numpy array: Laplacian matrix L = D - A
    """
    n = size1 + size2
    adj = np.zeros((n, n))
    
    # Generate edges within first group (0 to size1-1)
    for i in range(size1):
        for j in range(i+1, size1):
            if np.random.random() < prob1:
                adj[i, j] = adj[j, i] = 1
    
    # Generate edges within second group (size1 to n-1)
    for i in range(size1, n):
        for j in range(i+1, n):
            if np.random.random() < prob2:
                adj[i, j] = adj[j, i] = 1
    
    # Generate edges between groups
    for i in range(size1):
        for j in range(size1, n):
            if np.random.random() < prob_between:
                adj[i, j] = adj[j, i] = 1
    
    # Create Laplacian matrix L = D - A
    degree = np.sum(adj, axis=1)
    laplacian = np.diag(degree) - adj
    
    return laplacian

def generate_multi_group_laplacian(num_groups, group_size, prob_within, prob_between):
    """
    Generate a Laplacian matrix for multiple groups with uniform group sizes.
    
    Args:
        num_groups: int, number of groups
        group_size: int, number of nodes in each group
        prob_within: float, probability of edge within each group (0 to 1)
        prob_between: float, probability of edge between different groups (0 to 1)
    
    Returns:
        numpy array: Laplacian matrix L = D - A
    """
    
    # Validate inputs
    if num_groups <= 0 or group_size <= 0:
        raise ValueError("num_groups and group_size must be positive integers")
    if not (0 <= prob_within <= 1) or not (0 <= prob_between <= 1):
        raise ValueError("Probabilities must be between 0 and 1")
    
    # Calculate total number of nodes
    total_nodes = num_groups * group_size
    
    # Initialize adjacency matrix
    adj = np.zeros((total_nodes, total_nodes))
    
    # Generate edges within each group
    for group_id in range(num_groups):
        start_idx = group_id * group_size
        end_idx = (group_id + 1) * group_size
        
        # Generate all possible pairs within this group
        for i in range(start_idx, end_idx):
            for j in range(i + 1, end_idx):
                if np.random.random() < prob_within:
                    adj[i, j] = adj[j, i] = 1
    
    # Generate edges between different groups
    for group1 in range(num_groups):
        for group2 in range(group1 + 1, num_groups):
            start1, end1 = group1 * group_size, (group1 + 1) * group_size
            start2, end2 = group2 * group_size, (group2 + 1) * group_size
            
            # Generate edges between group1 and group2
            for i in range(start1, end1):
                for j in range(start2, end2):
                    if np.random.random() < prob_between:
                        adj[i, j] = adj[j, i] = 1
    
    # Create Laplacian matrix L = D - A
    degree = np.sum(adj, axis=1)
    laplacian = np.diag(degree) - adj

    return laplacian

def generate_layers_groups_graph(
    num_supergroups=2,
    num_subgroups_per_supergroup=2,
    nodes_per_subgroup=5,
    p_intra_subgroup=0.8,
    p_intra_supergroup=0.3,
    p_inter_supergroup=0.05,
    seed = None
):
    """
    Generates a hierarchical adjacency matrix and its Laplacian.
    Structure: supergroup → subgroup → nodes.
    """
    if seed is None:
        np.random.seed(841)
    else:
        np.random.seed(seed)
        
    total_nodes = num_supergroups * num_subgroups_per_supergroup * nodes_per_subgroup
    A = np.zeros((total_nodes, total_nodes))

    def get_node_index(group, subgroup, node):
        return (group * num_subgroups_per_supergroup + subgroup) * nodes_per_subgroup + node

    # Intra-supergroup connections
    for g in range(num_supergroups):
        for sg1 in range(num_subgroups_per_supergroup):
            for sg2 in range(num_subgroups_per_supergroup):
                for n1 in range(nodes_per_subgroup):
                    for n2 in range(nodes_per_subgroup):
                        i = get_node_index(g, sg1, n1)
                        j = get_node_index(g, sg2, n2)
                        if i >= j:
                            continue
                        p = p_intra_subgroup if sg1 == sg2 else p_intra_supergroup
                        if np.random.rand() < p:
                            A[i, j] = A[j, i] = 1

    # Inter-supergroup connections
    for g1 in range(num_supergroups):
        for g2 in range(g1 + 1, num_supergroups):
            for sg1 in range(num_subgroups_per_supergroup):
                for sg2 in range(num_subgroups_per_supergroup):
                    for n1 in range(nodes_per_subgroup):
                        for n2 in range(nodes_per_subgroup):
                            i = get_node_index(g1, sg1, n1)
                            j = get_node_index(g2, sg2, n2)
                            if np.random.rand() < p_inter_supergroup:
                                A[i, j] = A[j, i] = 1

    degree = np.sum(A, axis=1)
    L = np.diag(degree) - A

    return L

def matrix_shuffle(matrix):
    random_order = random.sample(range(len(matrix)), len(matrix))
    permuted_lap = matrix[np.ix_(random_order, random_order)]
    return permuted_lap

def visualize_laplacian_matrix(laplacian_matrix, show = True):
    """
    Visualize a Laplacian matrix as a black and white image.
    
    Args:
        laplacian_matrix
    
    Returns:
        visualization of the connections
    """
    if sp.isspmatrix_csr(laplacian_matrix): laplacian_matrix = laplacian_matrix.toarray()
    # Filled the matrix's diagonal with 0
    # Create binary matrix: 1 for non-zero elements, 0 for zero elements
    binary_matrix = (laplacian_matrix != 0).astype(int)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(6,6))
    
    # Display the matrix: black for non-zero (1), white for zero (0)
    im = ax.imshow(binary_matrix, cmap='gray_r', interpolation='nearest')
    
    if show:
        # Show the plot
        plt.show()
    else:
        return fig
    
def combine_three_figures(fig1, fig2, fig3, titles=None):
    """
    Combine three existing figures into one figure with subplots.
    
    Args:
        fig1, fig2, fig3: matplotlib figure objects from visualize_laplacian_matrix
        titles: Optional list of titles for each subplot
    
    Returns:
        Combined figure
    """
    # Create new figure with 3 subplots
    combined_fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))
    
    # Get the image data from each figure
    img1 = fig1.axes[0].images[0].get_array()
    img2 = fig2.axes[0].images[0].get_array()
    img3 = fig3.axes[0].images[0].get_array()
    
    # Display in new subplots
    ax1.imshow(img1, cmap='gray_r', interpolation='nearest')
    ax2.imshow(img2, cmap='gray_r', interpolation='nearest')
    ax3.imshow(img3, cmap='gray_r', interpolation='nearest')
    
    # Add titles if provided
    if titles:
        ax1.set_title(titles[0])
        ax2.set_title(titles[1])
        ax3.set_title(titles[2])
    
    plt.show()
    # Close the original figures to free memory
    plt.close(fig1)
    plt.close(fig2)
    plt.close(fig3)
    

    return combined_fig

def grade_matrix(matrix):
    """
    Calculate grading energy: for all i>j, if matrix[i,j]=1, sum up (i-j)
    
    Args:
        matrix: 2D numpy array containing only 0s and -1s (ignore the diagonals)
    
    Returns:
        float: grading energy value
    """
    n = matrix.shape[0]
    i_indices, j_indices = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
    
    mask = i_indices > j_indices
    distance_matrix = j_indices - i_indices
    
    return int(np.sum(distance_matrix * mask * matrix))


    
    
    