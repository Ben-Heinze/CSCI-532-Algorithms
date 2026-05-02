import random

def generate_graph(n, density=0.1, seed=7, cap_range=(1, 20)):
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
    print(nodes)
    random.shuffle(nodes)
    print(nodes)

    # Force source at start, sink at end (puts node 0 at start of list, and sink (n-1) at end of list)
    if nodes[0] != source:
        i = nodes.index(source)
        nodes[0], nodes[i] = nodes[i], nodes[0]
    if nodes[-1] != sink:
        i = nodes.index(sink)
        nodes[-1], nodes[i] = nodes[i], nodes[-1]
    print(nodes)

    # Create a path through all nodes
    for i in range(n - 1):
        u, v = nodes[i], nodes[i + 1]
        edges[(u, v)] = random.randint(*cap_range)

    # --- Step 2: add remaining edges based on density ---
    max_edges = n * (n - 1)  # no self-loops
    target_edges = int(density * max_edges)

    # Ensure at least the path edges are counted
    target_edges = max(target_edges, n - 1)

    while len(edges) < target_edges:
        u = random.randint(0, n - 1)
        v = random.randint(0, n - 1)

        if u != v and (u, v) not in edges:
            edges[(u, v)] = random.randint(*cap_range)

    print(f'Edges: {edges}\nsource:{source}\nsink:{sink}')
    return edges, source, sink

import time
from docplex.mp.model import Model

def solve_with_cplex(graph, source, sink):
    """
    Solves max flow using CPLEX.

    Parameters:
        graph (dict): {(u, v): capacity}
        source (int)
        sink (int)

    Returns:
        flow_value (float)
        setup_time (float)
        solve_time (float)
        total_time (float)
    """

    # -----------------------
    # Model setup
    # -----------------------
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

    # -----------------------
    # Solve
    # -----------------------
    start_solve = time.perf_counter()
    solution = mdl.solve()
    end_solve = time.perf_counter()

    if solution is None:
        return None, None, None, None

    flow_value = solution.objective_value

    setup_time = end_setup - start_setup
    solve_time = end_solve - start_solve
    total_time = setup_time + solve_time

    return flow_value, setup_time, solve_time, total_time

edges, s, t = generate_graph(n=10, density=0.1)

flow, setup_t, solve_t, total_t = solve_with_cplex(edges, s, t)

print("Max flow:", flow)
print("Setup time:", setup_t)
print("Solve time:", solve_t)
print("Total time:", total_t)
