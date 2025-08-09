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
    shuf_energy = grade_matrix(L)
    ordered_energy = grade_matrix(ordered_L)
    if show: print(f"Total size {len(oL)}, took {duration:.2f}s, peak memory {peak_memory:.2f}MB, energy ratio {100 * ordered_energy/shuf_energy:.2f}%({100 * ordered_energy/orig_energy:.2f}%)")
    return duration, peak_memory, ordered_energy/shuf_energy, ordered_energy/orig_energy

import csv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def parallal_choices_test():
    # Prepare CSV file
    csv_filename = 'parallel_test_results.csv'
    print(f"Starting parallel choices test - results will be saved to {csv_filename}")
    
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write header
        writer.writerow(['matrix_type', 'total_nodes', 'a_thre10', 'b_thre10', 'c_thre10', 'd_thre10', 
                        'a_thre1', 'b_thre1', 'c_thre1', 'd_thre1'])
        
        for i in range(1, 51):
            print(f"Processing iteration {i}/50 (total_nodes = {i*10*10})")
            
            # Matrix 1: varying supergroups
            print(f"  Matrix 1 - supergroups: {i}×10×10")
            L = generate_layers_groups_graph(
                num_supergroups=i,
                num_subgroups_per_supergroup=10,
                nodes_per_subgroup=10,
                p_intra_subgroup=0.8,
                p_intra_supergroup=0.3,
                p_inter_supergroup=0.05,
            )
            total_nodes = i * 10 * 10
            
            a1, b1, c1, d1 = test_duration_memory(L, thre=10)
            a2, b2, c2, d2 = test_duration_memory(L, thre=1)
            print(f"    thre=10: a={a1:.3f}, b={b1:.3f}")
            print(f"    thre=1:  a={a2:.3f}, b={b2:.3f}")
            
            writer.writerow(['supergroups', total_nodes, a1, b1, c1, d1, a2, b2, c2, d2])
            
            # Matrix 2: varying subgroups
            print(f"  Matrix 2 - subgroups: 1×{i*10}×10")
            L2 = generate_layers_groups_graph(
                num_supergroups=1,
                num_subgroups_per_supergroup=i*10,
                nodes_per_subgroup=10,
                p_intra_subgroup=0.8,
                p_intra_supergroup=0.3,
                p_inter_supergroup=0.05,
            )
            
            a1, b1, c1, d1 = test_duration_memory(L2, thre=10)
            a2, b2, c2, d2 = test_duration_memory(L2, thre=1)
            print(f"    thre=10: a={a1:.3f}, b={b1:.3f}")
            print(f"    thre=1:  a={a2:.3f}, b={b2:.3f}")
            
            writer.writerow(['subgroups', total_nodes, a1, b1, c1, d1, a2, b2, c2, d2])
            
            # Matrix 3: varying nodes per subgroup
            print(f"  Matrix 3 - nodes_per_sub: 1×10×{i*10}")
            L3 = generate_layers_groups_graph(
                num_supergroups=1,
                num_subgroups_per_supergroup=10,
                nodes_per_subgroup=i*10,
                p_intra_subgroup=0.8,
                p_intra_supergroup=0.3,
                p_inter_supergroup=0.05,
            )
            
            a1, b1, c1, d1 = test_duration_memory(L3, thre=10)
            a2, b2, c2, d2 = test_duration_memory(L3, thre=1)
            print(f"    thre=10: a={a1:.3f}, b={b1:.3f}")
            print(f"    thre=1:  a={a2:.3f}, b={b2:.3f}")
            
            writer.writerow(['nodes_per_sub', total_nodes, a1, b1, c1, d1, a2, b2, c2, d2])
            
            if i % 10 == 0:
                print(f"  ✓ Completed {i}/50 iterations")
    
    print(f"✓ Test completed! Results saved to {csv_filename}")

def read_csv_to_arrays(csv_filename='parallel_test_results.csv'):
    """Read CSV data into separate numpy arrays for each matrix type and threshold"""
    data = {
        'supergroups': {'total_nodes': [], 'a_thre10': [], 'b_thre10': [], 'c_thre10': [], 'd_thre10': [],
                       'a_thre1': [], 'b_thre1': [], 'c_thre1': [], 'd_thre1': []},
        'subgroups': {'total_nodes': [], 'a_thre10': [], 'b_thre10': [], 'c_thre10': [], 'd_thre10': [],
                     'a_thre1': [], 'b_thre1': [], 'c_thre1': [], 'd_thre1': []},
        'nodes_per_sub': {'total_nodes': [], 'a_thre10': [], 'b_thre10': [], 'c_thre10': [], 'd_thre10': [],
                         'a_thre1': [], 'b_thre1': [], 'c_thre1': [], 'd_thre1': []}
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
    """Create 4 graphs with the specified requirements"""
    data = read_csv_to_arrays(csv_filename)
    
    # Define colors for each matrix type
    colors = {'supergroups': 'blue', 'subgroups': 'red', 'nodes_per_sub': 'green'}
    
    # Create 4 subplots for a, b, c, d values
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    metrics = ['a', 'b', 'c', 'd']
    titles = ['Metric A', 'Metric B', 'Metric C', 'Metric D']
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        for matrix_type in ['supergroups', 'subgroups', 'nodes_per_sub']:
            x_data = data[matrix_type]['total_nodes']
            
            # Plot thre=1 (solid line)
            y_data_thre1 = data[matrix_type][f'{metric}_thre1']
            ax.plot(x_data, y_data_thre1, color=colors[matrix_type], 
                   linestyle='-', label=f'{matrix_type} (thre=1)', linewidth=2)
            
            # Plot thre=10 (dotted line, same color)
            y_data_thre10 = data[matrix_type][f'{metric}_thre10']
            ax.plot(x_data, y_data_thre10, color=colors[matrix_type], 
                   linestyle=':', label=f'{matrix_type} (thre=10)', linewidth=2)
        
        ax.set_xlabel('Total Nodes (num_sup × num_sub × nodes_per_sub)')
        ax.set_ylabel(f'{titles[idx]} Value')
        ax.set_title(f'{titles[idx]} vs Total Nodes')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('parallel_test_results.png', dpi=300, bbox_inches='tight')
    plt.show()

# Example usage:
# First run the test
parallal_choices_test()
# Then create the plots
plot_results()