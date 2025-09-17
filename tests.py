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

import numpy as np

def test_duration_memory(
    oL,
    thre=None,
    show=False,
    use_parallel=False,
    workers=None,
    max_parallel_depth=None,
    stru = "dense"
):
    """
    行为和你现有的 test_duration_memory 一致，但多了 use_parallel 与并行参数。
    内部用同样的 pipeline：matrix_shuffle -> build tree -> get_order -> 计算 grade ratio。
    差别只在于：根据 use_parallel 选择 treebuilder 或 treebuilder_parallel。
    """
    # 1) 乱序
    L = matrix_shuffle(oL)

    T, duration, peak_memory = measure_time_and_memory(treebuilder, L, thre, None, stru, use_parallel)

    # 4) 取顺序并重排
    order = T.get_order()
    ordered_L = L[np.ix_(order, order)]

    # 5) 可视化（保持你原逻辑）
    if show:
        fig1 = visualize_laplacian_matrix(oL, show=False)
        fig2 = visualize_laplacian_matrix(L, show=False)
        fig3 = visualize_laplacian_matrix(ordered_L, show=False)
        combine_three_figures(fig1, fig2, fig3,
                              titles=['Original', 'Shuffled', 'Restored'])

    # 6) 正确口径的 energy ratio
    orig_energy = grade_matrix(oL)
    ordered_energy = grade_matrix(ordered_L)
    ratio = ordered_energy / orig_energy

    return duration, peak_memory, ratio

import csv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
def run_one_graph_test(sup: int, sub: int, node: int, thre: int, parallel: bool = False, structure: str = "dense"):
    """
    只跑一次：构图 -> 两次测量(thre=node 与 thre=1)。
    不打印，不写文件；返回一个 dict，外层随意使用。
    """
    # 构建一次图
    L = generate_layers_groups_graph(
        num_supergroups=sup,
        num_subgroups_per_supergroup=sub,
        nodes_per_subgroup=node,
        p_intra_subgroup=0.8,
        p_intra_supergroup=0.3,
        p_inter_supergroup=0.05,
    )
    
    a, b, c = test_duration_memory(L, thre=thre, use_parallel = parallel, stru = structure)

    return {
        "duration": a,
        "memory": b,
        "ratio": c
    }

def parallal_choices_test(iter: int, sup: int, sub: int, node: int,
                          csv_filename: str = "parallel_test_results.csv"):
    """
    外层循环驱动（单次阈值版本）：
      - 对每个 i=1..iter，分别放大：
        1) supergroups:  (i*sup, sub, node)         -> 'supergroups'
        2) subgroups:    (sup, i*sub, node)         -> 'subgroups'
        3) nodes_per_sub:(sup, sub, i*node)         -> 'nodes_per_sub'
    现在 run_one_graph_test 只返回一次结果: {'duration','memory','ratio'}。
    CSV 结构改为：
        sup, sub, node, matrix_type, total_nodes, duration, memory, ratio
    """
    print(f"Starting parallel choices test - results will be saved to {csv_filename}")

    need_header = (not os.path.exists(csv_filename)) or os.path.getsize(csv_filename) == 0
    with open(csv_filename, "a", newline="") as f:
        writer = csv.writer(f)
        if need_header:
            writer.writerow([
                "sup", "sub", "node",
                "matrix_type", "total_nodes",
                "duration", "memory", "ratio",
            ])

        for i in range(1, iter + 1):
            base_total = sup * sub * node
            print(f"Processing iteration {i}/{iter} (total_nodes base = {base_total}, scale = {i})")

            # 1) 放大 sup
            sup_i, sub_i, node_i = i * sup, sub, node
            res = run_one_graph_test(sup_i, sub_i, node_i)
            total_nodes = sup_i * sub_i * node_i
            print(f"  Matrix 1 - supergroups: {sup_i}×{sub_i}×{node_i}")
            print(f"    duration={res['duration']:.3f}s, peak memory={res['memory']:.3f}Mb, ratio={100*res['ratio']:.1f}%")
            writer.writerow([sup_i, sub_i, node_i, "supergroups", total_nodes,
                             res["duration"], res["memory"], res["ratio"]])

            # 2) 放大 sub
            sup_i, sub_i, node_i = sup, i * sub, node
            res = run_one_graph_test(sup_i, sub_i, node_i)
            total_nodes = sup_i * sub_i * node_i
            print(f"  Matrix 2 - subgroups: {sup_i}×{sub_i}×{node_i}")
            print(f"    duration={res['duration']:.3f}s, peak memory={res['memory']:.3f}Mb, ratio={100*res['ratio']:.1f}%")
            writer.writerow([sup_i, sub_i, node_i, "subgroups", total_nodes,
                             res["duration"], res["memory"], res["ratio"]])

            # 3) 放大 node
            sup_i, sub_i, node_i = sup, sub, i * node
            res = run_one_graph_test(sup_i, sub_i, node_i)
            total_nodes = sup_i * sub_i * node_i
            print(f"  Matrix 3 - nodes_per_sub: {sup_i}×{sub_i}×{node_i}")
            print(f"    duration={res['duration']:.3f}s, peak memory={res['memory']:.3f}Mb, ratio={100*res['ratio']:.1f}%")
            writer.writerow([sup_i, sub_i, node_i, "nodes_per_sub", total_nodes,
                             res["duration"], res["memory"], res["ratio"]])

            if i % 10 == 0:
                print(f"  ✓ Completed {i}/{iter} iterations")

    print(f"✓ Test completed! Results saved to {csv_filename}")


def read_csv_to_arrays(csv_filename='parallel_test_results.csv'):
    """
    读取新的 CSV 结构为 numpy 数组：
      data[matrix_type] 包含：
        'sup','sub','node','total_nodes','duration','memory','ratio'
    """
    data = {
        'supergroups': {'sup': [], 'sub': [], 'node': [], 'total_nodes': [],
                        'duration': [], 'memory': [], 'ratio': []},
        'subgroups': {'sup': [], 'sub': [], 'node': [], 'total_nodes': [],
                      'duration': [], 'memory': [], 'ratio': []},
        'nodes_per_sub': {'sup': [], 'sub': [], 'node': [], 'total_nodes': [],
                          'duration': [], 'memory': [], 'ratio': []}
    }

    with open(csv_filename, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        # 断言新表头存在
        required_cols = {"sup","sub","node","matrix_type","total_nodes","duration","memory","ratio"}
        missing = required_cols - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV header missing expected columns: {missing}. "
                             f"Make sure you're using the NEW format.")

        for row in reader:
            mtype = row['matrix_type']
            if mtype not in data:
                # 忽略未知类型（或可选择 raise）
                continue
            data[mtype]['sup'].append(int(float(row['sup'])))
            data[mtype]['sub'].append(int(float(row['sub'])))
            data[mtype]['node'].append(int(float(row['node'])))
            data[mtype]['total_nodes'].append(int(float(row['total_nodes'])))
            data[mtype]['duration'].append(float(row['duration']))
            data[mtype]['memory'].append(float(row['memory']))
            data[mtype]['ratio'].append(float(row['ratio']))

    # 转 numpy array
    for mtype in data:
        for key in data[mtype]:
            # sup/sub/node/total_nodes 用整数更自然
            if key in ('sup','sub','node','total_nodes'):
                data[mtype][key] = np.array(data[mtype][key], dtype=int)
            else:
                data[mtype][key] = np.array(data[mtype][key], dtype=float)

    return data


def plot_results(csv_filename='parallel_test_results.csv'):
    """
    画 3 张子图：duration / memory / ratio（每个 matrix_type 一条线）。
    x 轴为 total_nodes；自动对每类按 x 排序，避免连线乱跳。
    """
    data = read_csv_to_arrays(csv_filename)

    colors = {'supergroups': 'blue', 'subgroups': 'red', 'nodes_per_sub': 'green'}
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    metrics = ['duration', 'memory', 'ratio']
    titles = ['Duration (seconds)', 'Peak Memory (MB)', 'Ratio']

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        for mtype in ['supergroups', 'subgroups', 'nodes_per_sub']:
            x = data[mtype]['total_nodes']
            y = data[mtype][metric]

            # 按 x 排序之后再画
            if x.size > 0:
                order = np.argsort(x)
                ax.plot(x[order], y[order],
                        color=colors[mtype], linestyle='-',
                        label=mtype, linewidth=2)

        ax.set_xlabel('Total Nodes (sup × sub × node)')
        ax.set_ylabel(titles[idx])
        ax.set_title(f'{titles[idx]} vs Total Nodes')
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()
    plt.savefig('parallel_test_results.png', dpi=300, bbox_inches='tight')
    plt.show()

L = generate_layers_groups_graph(2,20,30,0.8,0.3,0.05)
print(test_duration_memory(L,thre=1,show=False,use_parallel=True,stru = "dense"))
print(test_duration_memory(L,thre=1,show=False,use_parallel=False,stru = "dense"))
print(test_duration_memory(L,thre=1,show=False,use_parallel=False,stru = "sparse"))
# visualize_laplacian_matrix(L)
# print(measure_time_and_memory(eigsh,csr_matrix(L),k=2,which = 'SA')[1])
# print(measure_time_and_memory(eigsh,L,k=2,which = 'SA')[1])
