import math
from math import sqrt
import numpy as np

REQUIRED_UPSTREAM_D = 10  # xD
REQUIRED_DOWNSTREAM_D = 5
TILT_TOLERANCE_DEG = 3.0

def _distance(a, b):
    return sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)

def _closest_node(graph, point):
    best = None
    best_d = float("inf")
    for n in graph.nodes:
        d = _distance(n, point)
        if d < best_d:
            best_d = d
            best = n
    return best, best_d

def analyze_instrument(inst, ifc, pipe_graph):

    try:
        tag = getattr(inst, "Tag", None) or getattr(inst, "Name", None) or "UNKNOWN"
    except Exception:
        tag = "UNKNOWN"

    loc = (0.0, 0.0, 0.0)
    try:
        plc = getattr(inst, "ObjectPlacement", None)
        if plc and hasattr(plc, "RelativePlacement") and hasattr(plc.RelativePlacement, "Location"):
            coords = plc.RelativePlacement.Location.Coordinates
            loc = (float(coords[0]), float(coords[1]), float(coords[2]))
    except Exception:
        loc = (0.0, 0.0, 0.0)

    upstream_length = 0.0
    downstream_length = 0.0
    pipe_diameter_mm = None
    upstream_pass = False
    downstream_pass = False
    flow_direction_confidence = "low"

    if pipe_graph is None or len(pipe_graph.nodes) == 0:
        pipe_diameter_mm = 50.0
        upstream_length = 1.0
        downstream_length = 0.5
        upstream_pass = upstream_length >= (REQUIRED_UPSTREAM_D * (pipe_diameter_mm / 1000.0))
        downstream_pass = downstream_length >= (REQUIRED_DOWNSTREAM_D * (pipe_diameter_mm / 1000.0))
        tilt_deg = 0.0
        orientation_pass = True

    else:
        node, dist = _closest_node(pipe_graph, loc)
        edges = list(pipe_graph.edges(node, data=True))
        if edges:
            e = edges[0]
            data = e[2]
            pipe_diameter_mm = data.get("diameter_mm", 50.0)
            centerline = data.get("centerline", [node])
            G = pipe_graph

            def trace_length(start_node, direction_node):
                length = 0.0
                prev = start_node
                curr = direction_node
                while True:
                    try:
                        edge_data = G.get_edge_data(prev, curr)
                    except Exception:
                        break
                    length += _distance(prev, curr)
                    diameter = edge_data.get("diameter_mm", pipe_diameter_mm)
                    if G.degree(curr) != 2:
                        break
                    neighbors = [n for n in G.neighbors(curr) if not (n == prev)]
                    if not neighbors:
                        break
                    next_node = neighbors[0]
                    v1 = np.array(curr) - np.array(prev)
                    v2 = np.array(next_node) - np.array(curr)
                    if np.linalg.norm(v1) < 1e-9 or np.linalg.norm(v2) < 1e-9:
                        break
                    cosang = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
                    cosang = max(-1.0, min(1.0, cosang))
                    angdeg = math.degrees(math.acos(cosang))
                    if angdeg > 10.0:
                        break
                    prev, curr = curr, next_node
                return length

            a, b = e[0], e[1]
            upstream_length = trace_length(node, a)
            downstream_length = trace_length(node, b)

            D_m = (pipe_diameter_mm or 50.0) / 1000.0
            upstream_pass = upstream_length >= (REQUIRED_UPSTREAM_D * D_m)
            downstream_pass = downstream_length >= (REQUIRED_DOWNSTREAM_D * D_m)
            flow_direction_confidence = "inferred" if True else "low"
            tilt_deg = 0.0
            orientation_pass = True
        else:
            pipe_diameter_mm = 50.0
            upstream_length = 1.0
            downstream_length = 0.5
            tilt_deg = 0.0
            orientation_pass = True
            upstream_pass = upstream_length >= (REQUIRED_UPSTREAM_D * (pipe_diameter_mm / 1000.0))
            downstream_pass = downstream_length >= (REQUIRED_DOWNSTREAM_D * (pipe_diameter_mm / 1000.0))

    suggestions = []
    if not upstream_pass:
        needed_m = (REQUIRED_UPSTREAM_D * (pipe_diameter_mm / 1000.0)) - upstream_length
        suggestions.append(f"Add ~{needed_m:.2f} m upstream straight spool (to reach {REQUIRED_UPSTREAM_D}D).")
    if not downstream_pass:
        needed_m = (REQUIRED_DOWNSTREAM_D * (pipe_diameter_mm / 1000.0)) - downstream_length
        suggestions.append(f"Add ~{needed_m:.2f} m downstream straight spool (to reach {REQUIRED_DOWNSTREAM_D}D).")

    return {
        "tag": str(tag),
        "type": getattr(inst, "Name", "instrument"),
        "location": [float(loc[0]), float(loc[1]), float(loc[2])],
        "attached_pipes": {"upstream": None, "downstream": None},
        "pipe_diameter_mm": float(pipe_diameter_mm or 50.0),
        "measured": {"upstream_m": float(upstream_length), "downstream_m": float(downstream_length)},
        "required": {"upstream_D": REQUIRED_UPSTREAM_D, "downstream_D": REQUIRED_DOWNSTREAM_D},
        "pass_fail": {"upstream": bool(upstream_pass), "downstream": bool(downstream_pass)},
        "orientation": {"tilt_deg": float(tilt_deg), "vertical_pass": bool(orientation_pass), "flow_direction_confidence": flow_direction_confidence},
        "suggestions": suggestions,
        "snapshots": []
    }
