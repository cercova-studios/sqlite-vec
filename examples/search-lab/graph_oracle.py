"""Test reference for finite directed graphs, independent of SQLite and native BFS.

Uses Floyd-Warshall all-pairs relaxation, deliberately unsuitable for large graphs.
Native helpers have no catalog: even an absent anchor has distance zero to itself.
"""


def minimum_hops(edges, anchors, max_depth=None):
    assert max_depth is None or (type(max_depth) is int and max_depth >= 0)
    anchors = set(anchors)
    pairs = [(s, d) for s, d in edges if s is not None and d is not None]
    vertices = anchors | {v for pair in pairs for v in pair}
    assert all(type(v) is int and -(2**63) <= v < 2**63 for v in vertices)
    distance = {(v, v): 0 for v in vertices}
    for s, d in pairs:
        distance[s, d] = min(distance.get((s, d), 1), 1)
    infinity = float("inf")
    for pivot in vertices:
        for s in vertices:
            for d in vertices:
                candidate = distance.get((s, pivot), infinity) + distance.get((pivot, d), infinity)
                if candidate < distance.get((s, d), infinity):
                    distance[s, d] = candidate
    result = {}
    for v in vertices:
        hops = min((distance.get((s, v), infinity) for s in anchors), default=infinity)
        if hops != infinity and (max_depth is None or hops <= max_depth):
            result[v] = hops
    return result
