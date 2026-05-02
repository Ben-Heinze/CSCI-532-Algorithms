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