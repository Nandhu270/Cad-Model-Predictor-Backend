
import math
import networkx as nx

SNAP_TOLERANCE = 1e-3

def snap_point(pt, tol=SNAP_TOLERANCE):
    return (round(pt[0]/tol)*tol, round(pt[1]/tol)*tol, round(pt[2]/tol)*tol)

def build_pipe_graph(ifc):
    G = nx.Graph()
    try:
        for pipe in ifc.by_type("IfcPipeSegment"):
            start = None
            end = None
            diameter = None

            try:
                loc = getattr(pipe, "ObjectPlacement", None)
                if loc and hasattr(loc, "RelativePlacement") and hasattr(loc.RelativePlacement, "Location"):
                    p = loc.RelativePlacement.Location
                    start = (float(p.Coordinates[0]), float(p.Coordinates[1]), float(p.Coordinates[2]))
            except Exception:
                pass

            if start is None:
                continue

            end = (start[0] + 1.0, start[1], start[2])
            diameter = 50.0

            a = snap_point(start)
            b = snap_point(end)
            G.add_node(a)
            G.add_node(b)
            G.add_edge(a, b, pipe=pipe, diameter_mm=diameter, centerline=[a, b])
    except Exception:
        pass

    return G
