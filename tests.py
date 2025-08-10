import numpy as np
from scipy.sparse import linalg as LA
import time
import matplotlib.pyplot as plt

from SCOPE import *
from matrix_master import *

def test_bicut_accuracy(size1, size2, prob1, prob2, prob_between, num_tests=10):
    """
    Test the accuracy of spectral clustering on generated graphs.
    
    Returns:
        float: accuracy rate (0 to 1)
    """
    correct = 0
    
    for _ in range(num_tests):
        # Generate test graph
        L = generate_test_laplacian(size1, size2, prob1, prob2, prob_between)
        
        # Run clustering
        predicted_group, _ = bicut_group(L)
        predicted_group.sort()
        
        # True first group is [0, 1, ..., size1-1]
        true_group = list(range(size1))
        if list(predicted_group) == true_group:
            correct += 1
    
    return correct / num_tests

import tracemalloc
import numpy as np
import time

def measure_time_and_memory(func, *args, **kwargs):
    """
    Measures execution time and peak memory of a function call.
    Returns (result, time_in_seconds, memory_in_MB).
    """
    tracemalloc.start()
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return result, end_time - start_time, peak/2**20

def test_duration_memory(oL, thre = None, show = False):
    L = matrix_shuffle(oL)
    T, duration, peak_memory = measure_time_and_memory(treebuilder, L , thre)
    order = T.get_order()
    ordered_L = L[np.ix_(order, order)]
    if show:
        fig1 = visualize_laplacian_matrix(oL, show=False)
        fig2 = visualize_laplacian_matrix(L, show=False)
        fig3 = visualize_laplacian_matrix(ordered_L, show=False)
        combine_three_figures(fig1, fig2, fig3, 
                            titles=['Original', 'Shuffled', 'Restored'])
    orig_energy = grade_matrix(oL)
    ordered_energy = grade_matrix(ordered_L)
    if show: print(f"Total size {len(oL)}, took {duration:.2f}s, peak memory {peak_memory:.2f}MB, energy ratio {100 * ordered_energy/orig_energy:.2f}%")
    return duration, peak_memory, ordered_energy/orig_energy

import csv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def parallal_choices_test(iter,sup,sub,node):
    # Prepare CSV file
    csv_filename = 'parallel_test_results.csv'
    print(f"Starting parallel choices test - results will be saved to {csv_filename}")
    
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write header
        writer.writerow(['matrix_type', 'total_nodes', 'duration_thre', 'memory_thre', 'ratio_thre', 
                        'duration', 'memory', 'ratio'])
        
        for i in range(1, iter+1):
            print(f"Processing iteration {i}/{iter} (total_nodes = {i*sup*sub*node})")
            
            # Matrix 1: varying supergroups
            print(f"  Matrix 1 - supergroups: {i*sup}×{sub}×{node}")
            L = generate_layers_groups_graph(
                num_supergroups=i*sup,
                num_subgroups_per_supergroup=sub,
                nodes_per_subgroup=node,
                p_intra_subgroup=0.8,
                p_intra_supergroup=0.3,
                p_inter_supergroup=0.05,
            )
            total_nodes = i *sup * sub * node
            
            a1, b1, c1 = test_duration_memory(L, thre=node)
            a2, b2, c2 = test_duration_memory(L, thre=1)
            print(f"    thre={node}: duration ={a1:.3f}s, peak memory={b1:.3f}Mb")
            print(f"    thre=1:  duration ={a2:.3f}s, peak memory={b2:.3f}Mb")
            
            writer.writerow(['supergroups', total_nodes, a1, b1, c1, a2, b2, c2])
            
            # Matrix 2: varying subgroups
            print(f"  Matrix 2 - subgroups: {sup}×{i*sub}×{node}")
            L2 = generate_layers_groups_graph(
                num_supergroups=sup,
                num_subgroups_per_supergroup=i*sub,
                nodes_per_subgroup=node,
                p_intra_subgroup=0.8,
                p_intra_supergroup=0.3,
                p_inter_supergroup=0.05,
            )
            
            a1, b1, c1 = test_duration_memory(L, thre=node)
            a2, b2, c2 = test_duration_memory(L, thre=1)
            print(f"    thre={node}: duration ={a1:.3f}s, peak memory={b1:.3f}Mb")
            print(f"    thre=1:  duration ={a2:.3f}s, peak memory={b2:.3f}Mb")
            
            writer.writerow(['subgroups', total_nodes, a1, b1, c1, a2, b2, c2])
            
            # Matrix 3: varying nodes per subgroup
            print(f"  Matrix 3 - nodes_per_sub: {sup}×{sub}×{i*node}")
            L3 = generate_layers_groups_graph(
                num_supergroups=sup,
                num_subgroups_per_supergroup=sub,
                nodes_per_subgroup=i*node,
                p_intra_subgroup=0.8,
                p_intra_supergroup=0.3,
                p_inter_supergroup=0.05,
            )
            
            a1, b1, c1 = test_duration_memory(L, thre=i*node)
            a2, b2, c2 = test_duration_memory(L, thre=1)
            print(f"    thre={i*node}: duration ={a1:.3f}s, peak memory={b1:.3f}Mb")
            print(f"    thre=1:  duration ={a2:.3f}s, peak memory={b2:.3f}Mb")
            
            writer.writerow(['nodes_per_sub', total_nodes, a1, b1, c1, a2, b2, c2])
            
            if i % 10 == 0:
                print(f"  ✓ Completed {i}/50 iterations")
    
    print(f"✓ Test completed! Results saved to {csv_filename}")

def read_csv_to_arrays(csv_filename='parallel_test_results.csv'):
    """Read CSV data into separate numpy arrays for each matrix type and threshold"""
    data = {
        'supergroups': {'total_nodes': [], 'duration_thre': [], 'memory_thre': [], 'ratio_thre': [],
                       'duration': [], 'memory': [], 'ratio': []},
        'subgroups': {'total_nodes': [], 'duration_thre': [], 'memory_thre': [], 'ratio_thre': [],
                     'duration': [], 'memory': [], 'ratio': []},
        'nodes_per_sub': {'total_nodes': [], 'duration_thre': [], 'memory_thre': [], 'ratio_thre': [],
                         'duration': [], 'memory': [], 'ratio': []}
    }
    
    with open(csv_filename, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            matrix_type = row['matrix_type']
            for key in data[matrix_type].keys():
                data[matrix_type][key].append(float(row[key]))
    
    # Convert to numpy arrays
    for matrix_type in data:
        for key in data[matrix_type]:
            data[matrix_type][key] = np.array(data[matrix_type][key])
    
    return data

def plot_results(csv_filename='parallel_test_results.csv'):
    """Create 3 graphs for duration, memory, and ratio with the specified requirements"""
    data = read_csv_to_arrays(csv_filename)
    
    # Define colors for each matrix type
    colors = {'supergroups': 'blue', 'subgroups': 'red', 'nodes_per_sub': 'green'}
    
    # Create 3 subplots for duration, memory, ratio
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    metrics = ['duration', 'memory', 'ratio']
    titles = ['Duration (seconds)', 'Peak Memory (MB)', 'Ratio']
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        for matrix_type in ['supergroups', 'subgroups', 'nodes_per_sub']:
            x_data = data[matrix_type]['total_nodes']
            
            # Plot thre=1 (solid line)
            y_data_thre1 = data[matrix_type][metric]
            ax.plot(x_data, y_data_thre1, color=colors[matrix_type], 
                   linestyle='-', label=f'{matrix_type} (thre=1)', linewidth=2)
            
            # Plot thre=variable (dotted line, same color)
            y_data_thre_var = data[matrix_type][f'{metric}_thre']
            ax.plot(x_data, y_data_thre_var, color=colors[matrix_type], 
                   linestyle=':', label=f'{matrix_type} (thre=variable)', linewidth=2)
        
        ax.set_xlabel('Total Nodes (num_sup × num_sub × nodes_per_sub)')
        ax.set_ylabel(titles[idx])
        ax.set_title(f'{titles[idx]} vs Total Nodes')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('parallel_test_results.png', dpi=300, bbox_inches='tight')
    plt.show()

# Example usage:
# First run the test with custom parameters
parallal_choices_test(iter=30, sup=1, sub=5, node=5)

# Then create the plots
plot_results()