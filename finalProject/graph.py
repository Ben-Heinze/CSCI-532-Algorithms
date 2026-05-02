import random
import time
from docplex.mp.model import Model
from collections import deque
import csv
import sys
sys.setrecursionlimit(10000)  # or higher
def generate_graph(n, density=0.1, seed=None, cap_range=(1, 20)):
    """
    Generate a directed graph

    Parameters:
        n (int): number of nodes
        density (float): fraction of possible edges to include (0 < density <= 1)
        seed (int, optional): random seed for reproducibility
        cap_range (tuple): (min_capacity, max_capacity)

    Returns:
        edges (dict): {(u, v): capacity}
        source (int): source node (0)
        sink (int): sink node (n-1)
    """
    if not (0 < density <= 1):
        raise ValueError("density must be in (0, 1]")

    if seed is not None:
        random.seed(seed)

    source = 0
    sink = n - 1

    edges = {}

    # --- Step 1: ensure at least one path from source to sink ---
    nodes = list(range(n))
    random.shuffle(nodes)

    # Force source at start, sink at end
    if nodes[0] != source:
        i = nodes.index(source)
        nodes[0], nodes[i] = nodes[i], nodes[0]
    if nodes[-1] != sink:
        i = nodes.index(sink)
        nodes[-1], nodes[i] = nodes[i], nodes[-1]

    # Create a path through all nodes
    for i in range(n - 1):
        u, v = nodes[i], nodes[i + 1]
        edges[(u, v)] = random.randint(*cap_range)

    # --- Step 2: add remaining edges based on density ---
    max_edges = n * (n - 1)
    true_max_edges = (n - 1) * (n - 2)  # accounts for sink/source restrictions
    target_edges = int(density * max_edges)
    target_edges = max(target_edges, n - 1)
    target_edges = min(target_edges, true_max_edges)  # <-- add this line

    while len(edges) < target_edges:
        u = random.randint(0, n - 1)
        v = random.randint(0, n - 1)

        if (u != v
                and (u, v) not in edges
                and u != sink        # no edges leaving sink
                and v != source):    # no edges entering source
            edges[(u, v)] = random.randint(*cap_range)

    return edges, source, sink



def edmonds_karp(edges, source, sink):
    start_setup = time.perf_counter()
    # Build the same adjacency structure
    graph = {}
    for (u, v), cap in edges.items():
        if u not in graph: graph[u] = {}
        if v not in graph: graph[v] = {}
        graph[u][v] = graph[u].get(v, 0) + cap
        graph.setdefault(v, {})[u] = graph[v].get(u, 0)

    def bfs():
        # Returns parent map if a path exists, else None
        visited = {source}
        parent = {source: None}
        queue = deque([source])
        while queue:
            u = queue.popleft()
            for v, cap in graph[u].items():
                if v not in visited and cap > 0:
                    visited.add(v)
                    parent[v] = u
                    if v == sink:
                        return parent
                    queue.append(v)
        return None
    end_setup = time.perf_counter()

    start_solve = time.perf_counter()
    max_flow = 0
    while True:
        parent = bfs()
        if parent is None:
            break
        # Trace path back from sink to find bottleneck capacity
        path_flow = float('inf')
        v = sink
        while v != source:
            u = parent[v]
            path_flow = min(path_flow, graph[u][v])
            v = u
        # Update residual capacities along the path
        v = sink
        while v != source:
            u = parent[v]
            graph[u][v] -= path_flow
            graph[v][u] = graph[v].get(u, 0) + path_flow
            v = u
        max_flow += path_flow
    end_solve = time.perf_counter()

    solve_time = end_solve - start_solve
    setup_time = end_setup - start_setup
    total_time = solve_time + setup_time

    return max_flow, solve_time, setup_time, total_time


def solve_with_cplex(graph, source, sink):
    # Setup
    start_setup = time.perf_counter()
    mdl = Model(name="max_flow")

    # Create variables
    flow = {
        (u, v): mdl.continuous_var(lb=0, ub=cap, name=f"f_{u}_{v}")
        for (u, v), cap in graph.items()
    }

    # Collect nodes
    nodes = set()
    for u, v in graph:
        nodes.add(u)
        nodes.add(v)

    # Flow conservation constraints
    for node in nodes:
        if node == source or node == sink:
            continue
        inflow = mdl.sum(flow[(u, v)] for (u, v) in graph if v == node)
        outflow = mdl.sum(flow[(u, v)] for (u, v) in graph if u == node)
        mdl.add_constraint(inflow == outflow)

    # Objective: maximize flow into sink
    mdl.maximize(
        mdl.sum(flow[(u, v)] for (u, v) in graph if v == sink)
    )

    end_setup = time.perf_counter()

    # Solve
    start_solve = time.perf_counter()
    solution = mdl.solve()
    end_solve = time.perf_counter()

    if solution is None:
        return None, None, None, None

    flow_value = solution.objective_value

    setup_time = end_setup - start_setup
    solve_time = end_solve - start_solve
    total_time = setup_time + solve_time

    return flow_value, solve_time, setup_time, total_time

def ford_fulkerson(edges, source, sink):

    start_setup = time.perf_counter()
    # Build adjacency dict from your edges dict
    graph = {}
    for (u, v), cap in edges.items():
        if u not in graph: graph[u] = {}
        if v not in graph: graph[v]: graph[v] = {}
        graph[u][v] = graph[u].get(v, 0) + cap
        if v not in graph[u] or u not in graph[v]:
            graph.setdefault(v, {})[u] = graph[v].get(u, 0)  # reverse edge with 0 capacity

    def dfs(u, sink, visited, flow):
        if u == sink:
            return flow
        visited.add(u)
        for v, cap in graph[u].items():
            if v not in visited and cap > 0:
                result = dfs(v, sink, visited, min(flow, cap))
                if result > 0:
                    graph[u][v] -= result
                    graph[v][u] = graph[v].get(u, 0) + result
                    return result
        return 0
    end_setup = time.perf_counter()

    start_solve = time.perf_counter()
    max_flow = 0
    while True:
        visited = set()
        augment = dfs(source, sink, visited, float('inf'))
        if augment == 0:
            break
        max_flow += augment
    end_solve = time.perf_counter()

    setup_time = end_setup - start_setup
    solve_time = end_solve - start_solve
    total_time = setup_time + solve_time
    return max_flow, solve_time, setup_time, total_time


def main():
    density = round(random.random(), 1)
    print(density)
    trials = 5
    graph_sizes = [1000, 2500, 5000] #
    densities = [0.1, 0.3, 0.5, 0.7, 0.9, 1] 

    with open('results2.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["graph_size (n)", "density", "edge_count", "algorithm", "trial", "max_flow_result", "solve_time", "setup_time", "total_time"])
        # for every graph size
        for n in graph_sizes:
            for d in densities:
                for trial in range(1, trials+1): 
                    # gen graph
                    edges, s, t = generate_graph(n, density=d)
                    # 1. CPLEX
                    flow, solve_t, setup_t, total_t = solve_with_cplex(edges.copy(), s, t)
                    print([n, d, len(edges),  "cplex", trial, flow, solve_t, setup_t, total_t])
                    writer.writerow([n, d, len(edges), "cplex", trial, flow, solve_t, setup_t, total_t])
                    # 2. FF
                    ff_result, ff_solve_t, ff_setup_t, ff_total = ford_fulkerson(edges.copy(), s, t)
                    print([n, d, len(edges), "ford-fulkerson", trial, ff_result, ff_solve_t, ff_setup_t, ff_total])
                    writer.writerow([n, d, len(edges), "ford-fulkerson", trial, ff_result, ff_solve_t, ff_setup_t, ff_total])
                    # 3. EK
                    ek_result, ek_solve_t, ek_setup_t, ek_total_time  = edmonds_karp(edges.copy(), s, t)
                    print([n, d, len(edges), "Edmonds-Karp", trial, ek_result, ek_solve_t, ek_setup_t, ek_total_time])
                    writer.writerow([n, d, len(edges), "Edmonds-Karp", trial, ek_result, ek_solve_t, ek_setup_t, ek_total_time])
main()


