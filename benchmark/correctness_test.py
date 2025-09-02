from functools import lru_cache
import numpy as np
from tqdm import tqdm

P2C = -1
P2P =  0
C2P = +1

class TopoSim:
    def __init__(self, filename, break_tie_fn, prefer_change=False, exclude=set()):
        self.ngbrs = {}
        n_edge = 0

        for line in open(filename, "r").readlines():
            if line.startswith("#"): continue
            a, b, rel = line.strip().split("|")[:3]
            n_edge += 1

            to_add = (a not in exclude) and (b not in exclude)

            if a not in self.ngbrs:
                self.ngbrs[a] = {C2P:set(),P2P:set(),P2C:set()}
            if to_add:
                self.ngbrs[a][int(rel)].add(b)

            if b not in self.ngbrs:
                self.ngbrs[b] = {C2P:set(),P2P:set(),P2C:set()}
            if to_add:
                self.ngbrs[b][-int(rel)].add(a)

        self.break_tie_fn = break_tie_fn
        self.prefer_change = prefer_change

        print(f"load: {filename}")
        print(f"nodes: {len(self.ngbrs):,}, edges: {n_edge:,}")

    def get_ngbrs(self, asn, rel=[C2P, P2P, P2C]):
        if type(rel) is list:
            return set().union(*[self.ngbrs[asn][i] for i in rel])
        else:
            return self.ngbrs[asn][rel]

    def get_rel(self, a, b):
        for k, v in self.ngbrs[a].items():
            if b in v: return k
        return None

    @staticmethod
    def equally_best_nexthop(working_rib, best_from_rel=P2C):
        best_length = float("inf")
        ties = []
        for ngbr in working_rib.keys():
            as_path, from_rel = working_rib[ngbr]
            if from_rel == best_from_rel:
                length = len(as_path)
                if length == best_length:
                    ties.append(ngbr)
                elif length < best_length:
                    ties = [ngbr]
                    best_length = length
            elif from_rel > best_from_rel:
                ties = [ngbr]
                best_from_rel = from_rel
                best_length = len(as_path)
        return ties

    @staticmethod
    def states_update(routes, ribs, working_as, as_path, from_rel, break_tie_fn, prefer_change):
        # as_path should not contain working_as
        # from_rel is the rel from as_path[0] to working_as
        ngbr = as_path[0]
        rib = ribs.setdefault(working_as, dict())
        rib[ngbr] = [as_path, from_rel]

        updated = False
        if working_as not in routes:
            routes[working_as] = [as_path, from_rel] # first receive
            updated = True
        else:
            old_as_path, old_from_rel = routes[working_as]
            if old_as_path[0] == ngbr: # from the same ngbr
                if len(as_path) <= len(old_as_path):
                    # new route better than or equal to old one
                    routes[working_as] = [as_path, from_rel]
                    updated = True
                else: # select new route from rib
                    routes[working_as] = rib[break_tie_fn(
                        TopoSim.equally_best_nexthop(rib, best_from_rel=from_rel))]
                    updated = True
            else:
                if (from_rel > old_from_rel
                    or (from_rel == old_from_rel
                        and len(as_path) < len(old_as_path))):
                    # from different ngbrs and better than old one
                    routes[working_as] = [as_path, from_rel]
                    updated = True
                elif (prefer_change 
                    and from_rel == old_from_rel
                        and len(as_path) == len(old_as_path)):
                    # from different ngbrs and equal to old one
                    # and prefer to re-select given `prefer_change`
                    routes[working_as] = rib[break_tie_fn(
                        TopoSim.equally_best_nexthop(rib, best_from_rel=from_rel))]
                    updated = old_as_path != routes[working_as][0]
        return routes[working_as] if updated else None

    @lru_cache(maxsize=1024)
    def sim_all_routes_to(self, origin_as):
        ribs = dict()
        routes = {origin_as: [[], None]}

        init_path = [origin_as]
        queue = [[working_as, init_path, from_rel]
                for from_rel, ngbrs in self.ngbrs[origin_as].items()
                    for working_as in ngbrs]

        while queue:
            _working_as, _as_path, _from_rel = queue.pop(0)
            updated_route = TopoSim.states_update(routes, ribs, _working_as,
                    _as_path, _from_rel, self.break_tie_fn, self.prefer_change)
            if updated_route is not None:
                _as_path, _from_rel = updated_route
                next_path = [_working_as] + _as_path
                queue += [[working_as, next_path, from_rel]
                    for from_rel, ngbrs in self.ngbrs[_working_as].items()
                        if from_rel <= _from_rel and
                            not (from_rel == 0 and _from_rel == 0) # valley-free
                                for working_as in ngbrs
                                    if working_as not in _as_path] # avoid circle

        return routes

def check_correctness(as_rels, rmatrix, n_sample=100):
    # deterministic break-tie: prioritize lowest index
    def break_tie(ties):
        if len(ties) == 1: # branch ASes will go here
            return ties[0]
        ties_idx = [rmatrix.asn2idx(t) for t in ties]
        return ties[np.argmin(ties_idx)]

    # BGP simulator based on route propagation
    ts = TopoSim(as_rels, break_tie, prefer_change=True, exclude={})

    # check routes to random ASes
    for asn2 in tqdm(np.random.choice(list(ts.ngbrs.keys()), size=n_sample, replace=False)):
        all_routes_to_asn2 = ts.sim_all_routes_to(asn2)

        for asn1 in ts.ngbrs:
            if asn1 in all_routes_to_asn2: # reachable
                path, rel = all_routes_to_asn2[asn1]
                rel = P2C if rel is None else -rel # first-hop rel; None if asn1 == asn2
                length = len(path)
            else:
                path = None
                rel = None

            _rel, _length = rmatrix.get_state(asn1, asn2)
            _path = rmatrix.get_path(asn1, asn2)

            # state & path check
            try:
                assert _rel == rel
                if rel is not None:
                    assert length == _length
                    assert _length == len(_path)
                    assert _path is not None
                    assert _path == path
                else:
                    assert _path is None
            except AssertionError:
                print(
                    f"{asn1} -> {asn2} failed\n"\
                    f"  rel: {rel}\n"\
                    f" _rel: {_rel}\n"\
                    f" path: {path}\n"\
                    f"_path: {_path}"
                )
                raise

    print("Results correct.")
