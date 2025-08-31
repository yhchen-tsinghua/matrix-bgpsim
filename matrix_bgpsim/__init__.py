from multiprocessing import RawArray, Pool
from ctypes import c_ubyte, c_int
from collections.abc import Iterable, Set, Mapping, Callable, Sequence
from collections import namedtuple, defaultdict
import numpy as np
import lz4.frame
import pickle
import os

class RMatrix:
    # AS relationship notations
    # (adhere to CAIDA's norms)
    P2C = -1 # provider-to-customer
    P2P =  0 # peer-to-peer
    C2P = +1 # customer-to-provider

    # RelMap type definition
    RelMap = namedtuple("RMatrix.RelMap", [
        "P2P", # accessed by either index  0 or .P2P
        "C2P", # accessed by either index +1 or .C2P
        "P2C", # accessed by either index -1 or .P2C
    ])

    # BranchRoute type definition
    BranchRoute = namedtuple("RMatrix.BranchRoute", [
        "root",      # root AS where the branch is connected
        "next_hop",  # next-hop ASN to the root AS
        "length",    # number of hops to the root AS
        "branch_id", # to identify different branches
    ])

    # shared matrix for multiprocessing in runtime
    __shared_matrix__ = {
        # "state": writable, dtype uint8
        # "next_hop": (optional) writable, dtype int32
        # "link1": read-only, dtype uint8
        # "link2": read-only, dtype uint8
    }

    # TODO (future): add lock for thread safety?

    @staticmethod
    def caida_reader(fpath):
        with open(fpath, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith("#"):
                    continue
                asn1, asn2, rel = line.split("|")[:3]
                rel = int(rel)
                yield asn1, asn2, rel

    def __init__(self, input_rels, excluded=None):
        (
            # core AS information
            self.__idx2asn__,
            self.__asn2idx__,
            self.__idx2ngbrs__,

            # branch AS information
            self.__asn2brts__,

        ) = RMatrix.construct_topology(input_rels, excluded=excluded)

        # matrix for simulation (init on runtime)
        self.__state__ = None
        self.__next_hop__ = None

    @staticmethod
    def construct_topology(input_rels, excluded=None):
        # TODO (future): parallel simulation for disconnected sub-topologies
        # If the topology has several disconnected areas, assign each with a
        # single simluation task, so the matrix size can be greatly reduced.

        # construct AS relationships reader
        if isinstance(input_rels, (str, bytes, os.PathLike)):
            input_rels = RMatrix.caida_reader(input_rels)
        else:
            assert isinstance(input_rels, Iterable), \
                f"input_rels should be path-like or CAIDA reader: received {type(input_rels)}"

        # construct membership checker for excluded ASes
        if excluded is None or excluded is False:
            excluded_check = lambda item: False
        elif isinstance(excluded, (Set, Mapping)):
            excluded_check = lambda item: item in excluded
        elif isinstance(container, Sequence) and not isinstance(container, str):
            excluded_check = lambda item: item in set(container)
        else:
            assert isinstance(excluded, Callable), \
                f"excluded should be Callable or container for membership check: received {type(excluded)}"
            excluded_check = excluded

        # construct neighbors and edges
        asn2ngbrs = defaultdict(lambda: RMatrix.RelMap(set(), set(), set()))
        edges = list()

        for asn1, asn2, rel in input_rels:
            if excluded_check(asn1) or excluded_check(asn2): # ignore those excluded
                continue
            edges.append((asn1, asn2, rel))
            asn2ngbrs[asn1][+rel].add(asn2) # asn1 -> asn2: +rel
            asn2ngbrs[asn2][-rel].add(asn1) # asn2 -> asn1: -rel

        # find all branches
        asn2brts = dict() # branch AS -> branch route
        branch_id = 0
        for asn, (peers, providers, customers) in asn2ngbrs.items(): # see RelMap definition
            # search from stub/dangling AS
            if len(customers) != 0 or len(peers) != 0 or len(providers) > 1:
                continue 

            # TODO (future): optimization for stub-peering ASes
            # (i.e., those with no customer/provider but 1 peer)
            # Routing tables of stub-peering ASes can be directly derived from their peers,
            # so they can be excluded from the core network too for simulation efficiency.

            branch = []
            while (len(customers) <= 1
                    and len(peers) == 0
                        and len(providers) == 1): # search all the way up
                branch.append(asn)

                # go to the only provider
                asn, = providers 
                peers, providers, customers = asn2ngbrs[asn]

            # if this whole sub-tree is dangling (disconnected from the core network)
            if len(providers) == 0 and len(peers) == 0: 
                # it might still have multiple customers, thus forming a sub-tree,
                # and asn is the root (it could also be a single dangling AS).
                # Add a self-pointing branch route for it so it will not be included
                # in the core network later, which improves simulation efficiency.
                asn2brts[asn] = RMatrix.BranchRoute(
                        root=asn, next_hop=None, length=0, branch_id=None)

            for i, branch_as in enumerate(branch[::-1]):
                upstream = branch[-i] if i > 0 else asn
                asn2brts[branch_as] = RMatrix.BranchRoute(
                        root=asn, next_hop=upstream, length=i+1, branch_id=branch_id)

            # A branch route means: {branch_as} can follow {next_hop} all the way up
            # along the branch, and after {length} hops, it can finally reach a root
            # AS, which can be either a core AS or the root of a dangling sub-tree.
            branch_id += 1

        # construct core networks
        idx2asn = list()
        asn2idx = dict()
        idx2ngbrs = list()

        def init_core(asn):
            if asn not in asn2idx:
                asn2idx[asn] = len(idx2asn)
                idx2asn.append(asn)
                idx2ngbrs.append(RMatrix.RelMap(list(), list(), list()))
            return asn2idx[asn]

        num_core_edges = 0
        for asn1, asn2, rel in edges:
            # init index for core ASes
            idx1 = init_core(asn1) if asn1 not in asn2brts else None
            idx2 = init_core(asn2) if asn2 not in asn2brts else None

            if idx1 is None or idx2 is None:
                continue

            # add core edges
            idx2ngbrs[idx1][+rel].append(idx2)
            idx2ngbrs[idx2][-rel].append(idx1)
            num_core_edges += 1

        num_core_nodes = len(idx2asn)
        num_nodes = num_core_nodes + len(asn2brts)
        num_edges = len(edges)
        print(f"Topology constructed")
        print(f"load: {input_rels}")
        print(f"nodes: {num_nodes:,}")
        print(f"core nodes: {num_core_nodes:,} ({num_core_nodes/num_nodes:.2%})")
        print(f"edges: {num_edges:,}")
        print(f"core edges: {num_core_edges:,} ({num_core_edges/num_edges:.2%})")

        return idx2asn, asn2idx, idx2ngbrs, asn2brts

    def asn2idx(self, asn):
        return self.__asn2idx__[asn]

    def idx2asn(self, idx):
        return self.__idx2asn__[idx]

    def idx2ngbrs(self, idx):
        return self.__idx2ngbrs__[idx]

    def asn2brts(self, asn):
        return self.__asn2brts__[asn]

    def is_core_asn(self, asn):
        return asn in self.__asn2idx__

    def is_branch_asn(self, asn):
        return asn in self.__asn2brts__

    def has_asn(self, asn):
        return self.is_core_asn(asn) or self.is_branch_asn(asn)

    @staticmethod
    def __iterate_state_cpu__(worker_id, left, right, max_iter):
        shared = RMatrix.__shared_matrix__

        state = np.frombuffer(
                shared["state"], dtype=np.uint8).reshape(shared["shape"], order="F")
        link1 = np.frombuffer(
                shared["link1"], dtype=np.uint8).reshape(shared["shape"], order="C")
        link2 = np.frombuffer(
                shared["link2"], dtype=np.uint8).reshape(shared["shape"], order="C")

        tmp0, tmp1, tmp2 = np.empty((3, state.shape[0]), dtype=np.uint8)

        finish_flag = np.zeros(right-left, dtype=bool)

        for cur_iter in range(max_iter):
            for j, r_col in enumerate(state[:, left:right].T):
                if finish_flag[j]: continue
                finish = True
                j_actual = left+j
                # the most significant 2 bits (tmp0 = msb01)
                tmp0[:] = r_col & 0b11_000000 
                # the most significant 2 bits exchanged (tmp1 = msb10)
                tmp1[:] = ((tmp0 << 1) | (tmp0 >> 1)) & 0b11_000000
                # msb01 & msb10 (tmp0 = msb01&msb10)
                tmp0[:] &= tmp1
                # msb10 | r_col (msb10|r_col)
                tmp1[:] |= r_col
                for i, l_rows in enumerate(zip(link1, link2)):
                    if i == j_actual: continue
                    last = r_col[i]
                    l_row1, l_row2 = l_rows
                    tmp2 = (l_row1 & tmp0) | (l_row2 & tmp1)
                    r_col[i] = np.max(tmp2) - 1
                    if last != r_col[i]:
                        finish = False
                finish_flag[j] = finish

            print(f"Worker-{worker_id}: iteration {cur_iter} finished.")

            if finish_flag.all():
                break

    @staticmethod
    def __iterate_state_and_next_hop_cpu__(worker_id, left, right, max_iter):
        shared = RMatrix.__shared_matrix__

        state = np.frombuffer(
                shared["state"], dtype=np.uint8).reshape(shared["shape"], order="F")
        link1 = np.frombuffer(
                shared["link1"], dtype=np.uint8).reshape(shared["shape"], order="C")
        link2 = np.frombuffer(
                shared["link2"], dtype=np.uint8).reshape(shared["shape"], order="C")
        next_hop = np.frombuffer(
                shared["next_hop"], dtype=np.int32).reshape(shared["shape"], order="F")

        tmp0, tmp1, tmp2 = np.empty((3, state.shape[0]), dtype=np.uint8)

        finish_flag = np.zeros(right-left, dtype=bool)

        for cur_iter in range(max_iter):
            for j, r_col in enumerate(state[:, left:right].T):
                if finish_flag[j]: continue
                finish = True
                j_actual = left+j
                next_hop_col = next_hop[:, j_actual]
                # the most significant 2 bits (tmp0 = msb01)
                tmp0[:] = r_col & 0b11_000000 
                # the most significant 2 bits exchanged (tmp1 = msb10)
                tmp1[:] = ((tmp0 << 1) | (tmp0 >> 1)) & 0b11_000000
                # (tmp0 = msb01 & msb10)
                tmp0[:] &= tmp1
                # (tmp1 = msb10 | r_col)
                tmp1[:] |= r_col
                for i, l_rows in enumerate(zip(link1, link2)):
                    if i == j_actual: continue
                    last = r_col[i]
                    l_row1, l_row2 = l_rows
                    tmp2[:] = (l_row1 & tmp0) | (l_row2 & tmp1)
                    next_idx = np.argmax(tmp2)
                    next_hop_col[i] = next_idx
                    r_col[i] = tmp2[next_idx] - 1
                    if last != r_col[i]:
                        finish = False
                finish_flag[j] = finish

            print(f"Worker-{worker_id}: iteration {cur_iter} finished.")

            if finish_flag.all():
                break

        # where no route is available, the next-hop is manually set to -1
        next_hop[:, left:right][state[:, left:right] <= 0b00_111111] = -1

    def run(self, n_jobs=1, max_iter=32, save_next_hop=True):
        RMatrix.__cpu_runner__(self.__idx2ngbrs__, n_jobs, max_iter, save_next_hop)()
        self.__state__ = RMatrix.__shared_matrix__.pop("state", None)
        self.__next_hop__ = RMatrix.__shared_matrix__.pop("next_hop", None)
        RMatrix.__shared_matrix__ = {}
        return self

    @staticmethod
    def __cpu_runner__(idx2ngbrs, n_jobs, max_iter, save_next_hop):
        # init matrix
        size = len(idx2ngbrs)
        shape = (size, size)

        state = RawArray(c_ubyte, shape[0]*shape[1]) # shared raw array
        state_np = np.frombuffer(state, dtype=np.uint8).reshape(shape, order="F") # numpy interface
        state_np[:] = 0b00_111111
        for i, ngbrs in enumerate(idx2ngbrs):
            state_np[ngbrs.C2P, i] = 0b11_111110 # one-hop route to providers
            state_np[ngbrs.P2P, i] = 0b10_111110 # one-hop route to peers
            state_np[ngbrs.P2C, i] = 0b01_111110 # one-hop route to customers
        state_np[np.arange(size), np.arange(size)] = 0b11_111111 # self-pointing route
        print(f"state matrix constructed")

        link1 = RawArray(c_ubyte, shape[0]*shape[1])
        link1_np = np.frombuffer(link1, dtype=np.uint8).reshape(shape, order="C")
        for i, ngbrs in enumerate(idx2ngbrs):
            link1_np[i, ngbrs.P2C] = 0b11_000000
            link1_np[i, ngbrs.P2P] = 0b10_000000
            link1_np[i, ngbrs.C2P] = 0b01_000000
        print(f"link1 matrix constructed")

        link2 = RawArray(c_ubyte, shape[0]*shape[1])
        link2_np = np.frombuffer(link2, dtype=np.uint8).reshape(shape, order="C")
        link2_np[:] = 0b00_111111
        for i, ngbrs in enumerate(idx2ngbrs):
            link2_np[i, ngbrs.C2P] = 0b01_111111
        print(f"link2 matrix constructed")

        # TODO (future): try use c_uint16 for efficiency if core ASes are fewer than 65536
        if save_next_hop:
            next_hop = RawArray(c_int, shape[0]*shape[1])
            next_hop_np = np.frombuffer(next_hop, dtype=np.int32).reshape(shape, order="F")
            next_hop_np[:] = -1
            print(f"next_hop matrix constructed")

        # split for parallel tasks
        assert n_jobs >= 1
        split = np.linspace(0, size, n_jobs+1).astype(int)
        print(f"runner with {n_jobs} processes.")

        if save_next_hop:
            def runner():
                def initializer(state, link1, link2, next_hop, shape):
                    RMatrix.__shared_matrix__["state"] = state
                    RMatrix.__shared_matrix__["link1"] = link1
                    RMatrix.__shared_matrix__["link2"] = link2
                    RMatrix.__shared_matrix__["next_hop"] = next_hop
                    RMatrix.__shared_matrix__["shape"] = shape

                initargs = (state, link1, link2, next_hop, shape)

                params = zip(range(n_jobs), split[:-1], split[1:], [max_iter]*n_jobs)

                with Pool(processes=n_jobs, initializer=initializer,
                        initargs=initargs) as pool:
                    pool.starmap(RMatrix.__iterate_state_and_next_hop_cpu__, params)
        else:
            def runner():
                def initializer(state, link1, link2, shape):
                    RMatrix.__shared_matrix__["state"] = state
                    RMatrix.__shared_matrix__["link1"] = link1
                    RMatrix.__shared_matrix__["link2"] = link2
                    RMatrix.__shared_matrix__["shape"] = shape

                initargs = (state, link1, link2, shape)

                params = zip(range(n_jobs), split[:-1], split[1:], [max_iter]*n_jobs)

                with Pool(processes=n_jobs, initializer=initializer,
                        initargs=initargs) as pool:
                    pool.starmap(RMatrix.__iterate_state_cpu__, params)

        return runner

    def get_state(self, asn1, asn2):
        '''Return `(s_type, s_len)`
           s_type: One of the values from `None`, `C2P`, `P2P` and `P2C`.
                Return `P2C` if the queried `asn1` and `asn2` are the same.
           s_len: The length of the path if it exists, i.e., when `s_type`
                is not `None`. Otherwise, the meaning of `s_len` is undefined.
        '''
        assert self.has_asn(asn1), f"{asn1} is not in the topology"
        assert self.has_asn(asn2), f"{asn2} is not in the topology"
        assert self.__state__ is not None, f"run simulation to get state matrix first"

        def decode_state_from_matrix(asn1, asn2):
            s = self.__state__[self.asn2idx(asn1), self.asn2idx(asn2)]
            s_type = [None, RMatrix.C2P, RMatrix.P2P, RMatrix.P2C][s >> 6]
            s_len = 0b00_111111 - (s & 0b00_111111)
            return s_type, s_len

        if self.is_branch_asn(asn1): # asn1 is branch AS
            brts1 = self.asn2brts(asn1)
            if self.is_branch_asn(asn2): # asn2 is branch AS too
                brts2 = self.asn2brts(asn2)
                if brts1.root == brts2.root: # in the same sub-tree
                    if brts1.branch_id == brts2.branch_id: # in the same branch (or both are root)
                        s_type = RMatrix.C2P if brts1.length > brts2.length else RMatrix.P2C
                        s_len = abs(brts1.length - brts2.length)
                    else: # in different branches (or one is root)
                        s_type = RMatrix.C2P if brts1.length > 0 else RMatrix.P2C
                        s_len = brts1.length + brts2.length
                else: # in different sub-trees
                    if self.is_core_asn(brts1.root) and self.is_core_asn(brts2.root):
                        # both branches are connected to the core network
                        s_type, s_len = decode_state_from_matrix(brts1.root, brts2.root)
                        if s_type is not None:
                            s_type = RMatrix.C2P
                            s_len += brts1.length
                            s_len += brts2.length
                    else: # at least one sub-tree is disconnected from the core network
                        s_type = None
                        s_len = 0
            else: # asn2 is core AS
                if self.is_core_asn(brts1.root): # branch is connected to the core network
                    s_type, s_len = decode_state_from_matrix(brts1.root, asn2)
                    if s_type is not None:
                        s_type = RMatrix.C2P
                        s_len += brts1.length
                else: # sub-tree is disconnected from the core network
                    s_type = None
                    s_len = 0
        else: # asn1 is core AS
            if self.is_branch_asn(asn2): # asn2 is branch AS
                brts2 = self.asn2brts(asn2)
                if self.is_core_asn(brts2.root): # branch is connected to the core network
                    s_type, s_len = decode_state_from_matrix(asn1, brts2.root)
                    if s_type is not None:
                        s_len += brts2.length
                else: # sub-tree is disconnected from the core network
                    s_type = None
                    s_len = 0
            else: # asn2 is core AS too
                s_type, s_len = decode_state_from_matrix(asn1, asn2)

        return s_type, s_len

    def get_path(self, asn1, asn2):
        '''Return `path`
           path: A list of ASNs that form the AS-level path (i.e., AS_path)
               from `asn1` to `asn2`. `asn1` is not included, while `asn2`
               is always the tail of the list, if the path exists. Return
               `None` is the path doesn't exist. Return `[]` if the queired
               `asn1` and `asn2` are the same.
        '''
        assert self.has_asn(asn1), f"{asn1} is not in the topology"
        assert self.has_asn(asn2), f"{asn2} is not in the topology"
        assert self.__state__ is not None, f"run simulation to get state matrix first"
        assert self.__next_hop__ is not None, \
                f"run simulation with save_next_hop=True to get next_hop matrix first"

        def decode_path_from_matrix(asn1, asn2):
            src_idx = self.asn2idx(asn1)
            dst_idx = self.asn2idx(asn2)
            path = []
            while src_idx != dst_idx:
                src_idx = self.__next_hop__[src_idx, dst_idx]
                if src_idx == -1:
                    path = None
                    break
                path.append(self.idx2asn(src_idx))
            return path

        def up_branch_search(asn1, asn2):
            path = []
            traveler = asn1
            while traveler != asn2:
                traveler = self.asn2brts(traveler).next_hop
                path.append(traveler)
            return path

        def down_branch_search(asn1, asn2):
            path = []
            traveler = asn2
            while traveler != asn1:
                path.append(traveler) # append it first as it is reverse-searching
                traveler = self.asn2brts(traveler).next_hop
            return path[::-1] # reverse back

        if self.is_branch_asn(asn1): # asn1 is branch AS
            brts1 = self.asn2brts(asn1)
            if self.is_branch_asn(asn2): # asn2 is branch AS too
                brts2 = self.asn2brts(asn2)
                if brts1.root == brts2.root: # in the same sub-tree
                    if brts1.branch_id == brts2.branch_id: # in the same branch (or both are root)
                        if brts1.length > brts2.length: # asn1 is lower
                            path = up_branch_search(asn1, asn2)
                        else: # asn1 == asn2 or asn2 is lower
                            path = down_branch_search(asn1, asn2)
                    else: # in different branches (or one is root)
                        path = up_branch_search(asn1, brts1.root)
                        path += down_branch_search(brts2.root, asn2)
                else: # in differnet sub-trees
                    if self.is_core_asn(brts1.root) and self.is_core_asn(brts2.root):
                        # both branches are connected to the core network
                        path = decode_path_from_matrix(brts1.root, brts2.root)
                        if path is not None:
                            path = up_branch_search(asn1, brts1.root) + path
                            path += down_branch_search(brts2.root, asn2)
                    else: # at least one sub-tree is disconnected from the core network
                        path = None
            else: # asn2 is core AS
                if self.is_core_asn(brts1.root): # branch is connected to the core network
                    path = decode_path_from_matrix(brts1.root, asn2)
                    if path is not None:
                        path = up_branch_search(asn1, brts1.root) + path
                else: # sub-tree is disconnected from the core network
                    path = None
        else: # asn1 is core AS
            if self.is_branch_asn(asn2): # asn2 is branch AS
                brts2 = self.asn2brts(asn2)
                if self.is_core_asn(brts2.root): # branch is connected to the core network
                    path = decode_path_from_matrix(asn1, brts2.root)
                    if path is not None:
                        path += down_branch_search(brts2.root, asn2)
                else: # sub-tree is disconnected from the core network
                    path = None
            else: # asn2 is core AS too
                path = decode_path_from_matrix(asn1, asn2)

        return path

    def dump(self, fpath):
        pickle.dump(self, lz4.frame.open(fpath, "wb"), protocol=4)

    @staticmethod
    def load(fpath):
        return pickle.load(lz4.frame.open(fpath, "rb"))
