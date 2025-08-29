import numpy as np
from multiprocessing import RawArray, Pool
from ctypes import c_ubyte, c_int
import lz4.frame
import pickle

def lz4dump(obj, fpath):
    pickle.dump(obj, lz4.frame.open(fpath, "wb"), protocol=4)

def lz4load(fpath):
    return pickle.load(lz4.frame.open(fpath, "rb"))

class RMatrix:
    # AS relationship
    P2C = -1
    P2P =  0
    C2P = +1

    # non-branch AS information
    __idx2asn__ = []
    __asn2idx__ = {}
    __idx_ngbrs__ = []

    # branch_AS -> {(gateway_AS, distance, branch_id, upstream) ...}
    __branch2gateway__ = {}
    __gateway_weights__ = None
    __gateway_branches__ = None

    # shared matrix
    __shared_matrix__ = {
        # "state": writable, dtype uint8
        # "next_hop": writable, dtype int32
        # "link1": Read-only, dtype uint8
        # "link2": Read-only, dtype uint8
    }

    # class init done flag
    __init_done__ = False

    @staticmethod
    def asn2idx(asn):
        return RMatrix.__asn2idx__[asn]

    @staticmethod
    def idx2asn(idx):
        return RMatrix.__idx2asn__[idx]

    @staticmethod
    def idx_ngbrs(idx):
        return RMatrix.__idx_ngbrs__[idx]

    @staticmethod
    def asn_in_branch(asn):
        return asn in RMatrix.__branch2gateway__

    @staticmethod
    def get_gateway(asn):
        return asn if not RMatrix.asn_in_branch(asn) \
            else RMatrix.__branch2gateway__[asn][0]

    @staticmethod
    def gateway_weights():
        if RMatrix.__gateway_weights__ is None:
            weights = np.ones(len(RMatrix.__idx2asn__), dtype=np.int32)
            for gw,_,_,_ in RMatrix.__branch2gateway__.values():
                weights[RMatrix.asn2idx(gw)] += 1
            RMatrix.__gateway_weights__ = weights
        return RMatrix.__gateway_weights__

    @staticmethod
    def gateway_branches():
        if RMatrix.__gateway_branches__ is None:
            branches = [set() for _ in range(len(RMatrix.__idx2asn__))]
            for gw,_,bid,_ in RMatrix.__branch2gateway__.values():
                branches[RMatrix.asn2idx(gw)].add(bid)
            RMatrix.__gateway_branches__ = branches
        return RMatrix.__gateway_branches__

    @staticmethod
    def has_asn(asn):
        return asn in RMatrix.__asn2idx__ or asn in RMatrix.__branch2gateway__

    @staticmethod
    def init_class(as_rel_fpath):
        if RMatrix.__init_done__:
            print("Warning: class RMatrix re-inited.")
            RMatrix.__idx2asn__ = []
            RMatrix.__asn2idx__ = {}
            RMatrix.__idx_ngbrs__ = []
            RMatrix.__branch2gateway__ = {}
            RMatrix.__shared_matrix__ = {}

        ngbr_map = {}
        edges = []

        def get_ngbrs(asn):
            if asn not in ngbr_map:
                ngbr_map[asn] = {RMatrix.P2C: set(), RMatrix.P2P: set(), RMatrix.C2P: set()}
            return ngbr_map[asn]

        for l in open(as_rel_fpath, "r").readlines():
            if l.startswith("#"): continue
            a, b, r = l.strip().split("|")[:3]
            rel = int(r)

            edges.append((a, b, rel))

            get_ngbrs(a)[+rel].add(b)
            get_ngbrs(b)[-rel].add(a)

        def init_idx(asn):
            if asn not in RMatrix.__asn2idx__:
                RMatrix.__asn2idx__[asn] = len(RMatrix.__idx2asn__)
                RMatrix.__idx2asn__.append(asn)
                RMatrix.__idx_ngbrs__.append({RMatrix.P2C: [], RMatrix.P2P: [], RMatrix.C2P: []})
            return RMatrix.__asn2idx__[asn]

        branch_id = 0
        for asn, ngbrs in ngbr_map.items():
            degree = sum([len(s) for s in ngbrs.values()])

            if degree > 1: # search from stub/dangling AS
                continue

            branch = []
            while (len(ngbrs[RMatrix.P2C]) <= 1
                and len(ngbrs[RMatrix.P2P]) == 0
                    and len(ngbrs[RMatrix.C2P]) == 1):
                branch.append(asn)
                asn, = ngbrs[RMatrix.C2P]
                ngbrs = ngbr_map[asn]

            init_idx(asn) # init possibly dangling AS 
            # (note that gateway AS could be dangling after edge pruning)

            for i, branch_as in enumerate(branch[::-1]):
                upstream = branch[-i] if i > 0 else asn
                RMatrix.__branch2gateway__[branch_as] = (asn, i+1, branch_id, upstream)

            if branch: branch_id += 1

        e_core = 0
        for a, b, rel in edges:
            if RMatrix.asn_in_branch(a) or RMatrix.asn_in_branch(b):
                continue

            idx_a = init_idx(a)
            idx_b = init_idx(b)

            RMatrix.idx_ngbrs(idx_a)[+rel].append(idx_b)
            RMatrix.idx_ngbrs(idx_b)[-rel].append(idx_a)
            e_core += 1

        n_core = len(RMatrix.__idx2asn__)
        n = n_core + len(RMatrix.__branch2gateway__)
        e = len(edges)
        print(f"class RMatrix init..")
        print(f"load: {as_rel_fpath}")
        print(f"nodes: {n:,}")
        print(f"non-branch nodes: {n_core:,} ({n_core/n:.2%})")
        print(f"edges: {e:,}")
        print(f"non-branch edges: {e_core:,} ({e_core/e:.2%})")
        print(f"branches: {branch_id:,}")

        RMatrix.__init_done__ = True

    def __init__(self, exclude=set()):
        self.__exclude__ = set()
        self.__exclude_in_branch__ = set()
        for asn in exclude:
            if not RMatrix.has_asn(asn): continue
            if RMatrix.asn_in_branch(asn):
                self.__exclude_in_branch__.add(asn)
            else:
                self.__exclude__.add(asn)
        print(f"instance of RMatrix init: "
              f"{len(self.__exclude__)}/{len(self.__exclude_in_branch__)} ASes excluded.")

        # exlcude indexing
        self.__exclude_mask__ = np.zeros(len(RMatrix.__idx2asn__), dtype=bool)
        for asn in self.__exclude__:
            self.__exclude_mask__[RMatrix.asn2idx(asn)] = True
        self.__exclude_idx__, = np.where(self.__exclude_mask__)

        self.__state__ = None
        self.__next_hop__ = None

    @staticmethod
    def iterate(worker_id, left, right, max_iter):
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
    def iterate2(worker_id, left, right, max_iter):
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

        # manually set to -1
        next_hop[:, left:right][state[:, left:right] <= 0b00_111111] = -1 

    def run(self, n_jobs=1, max_iter=30, record_next_hop=False):
        f_size = len(RMatrix.__idx2asn__) # full size
        p_size = f_size - len(self.__exclude__) # partial size

        exclude_mask = self.__exclude_mask__
        exclude_idx = self.__exclude_idx__

        # remap indexing
        full2partial = np.arange(f_size)
        for i in exclude_idx:
            full2partial[i:] -= 1
        full2partial[exclude_mask] = -1

        idx_ngbrs = [dict() for _ in range(p_size)]
        for i, __ngbrs__ in enumerate(RMatrix.__idx_ngbrs__):
            i = full2partial[i]
            if i == -1: continue
            for rel, __ngbr_list__ in __ngbrs__.items():
                idx_ngbrs[i][rel] = np.setdiff1d(full2partial[__ngbr_list__], [-1])
        print(f"remap indexing done")

        # init matrix
        shape = (p_size, p_size)

        state = RawArray(c_ubyte, shape[0]*shape[1]) # shared
        state_np = np.frombuffer(state, dtype=np.uint8).reshape(shape, order="F")
        state_np[:] = 0b00_111111
        for i, ngbrs in enumerate(idx_ngbrs):
            state_np[ngbrs[RMatrix.C2P], i] = 0b11_111110
            state_np[ngbrs[RMatrix.P2P], i] = 0b10_111110
            state_np[ngbrs[RMatrix.P2C], i] = 0b01_111110
        state_np[np.arange(p_size), np.arange(p_size)] = 0b11_111111
        print(f"state matrix constructed")

        link1 = RawArray(c_ubyte, shape[0]*shape[1]) # shared
        link1_np = np.frombuffer(link1, dtype=np.uint8).reshape(shape, order="C")
        for i, ngbrs in enumerate(idx_ngbrs):
            link1_np[i, ngbrs[RMatrix.P2C]] = 0b11_000000
            link1_np[i, ngbrs[RMatrix.P2P]] = 0b10_000000
            link1_np[i, ngbrs[RMatrix.C2P]] = 0b01_000000
        print(f"link1 matrix constructed")

        link2 = RawArray(c_ubyte, shape[0]*shape[1]) # shared
        link2_np = np.frombuffer(link2, dtype=np.uint8).reshape(shape, order="C")
        link2_np[:] = 0b00_111111
        for i, ngbrs in enumerate(idx_ngbrs):
            link2_np[i, ngbrs[RMatrix.C2P]] = 0b01_111111
        print(f"link2 matrix constructed")

        if record_next_hop:
            next_hop = RawArray(c_int, shape[0]*shape[1])
            next_hop_np = np.frombuffer(next_hop, dtype=np.int32).reshape(shape, order="F")
            next_hop_np[:] = -1
            print(f"next_hop matrix constructed")

        # split for parallel tasks
        assert n_jobs >= 1
        split = np.linspace(0, p_size, n_jobs+1).astype(int)
        print(f"start running with {n_jobs} processes.")

        if record_next_hop:
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
                pool.starmap(RMatrix.iterate2, params)
            self.__state__ = RMatrix.expand_to_full_state(state_np, exclude_mask)
            self.__next_hop__ = RMatrix.expand_to_full_next_hop(next_hop_np, exclude_idx)
        else:
            def initializer(state, link1, link2, shape):
                RMatrix.__shared_matrix__["state"] = state
                RMatrix.__shared_matrix__["link1"] = link1
                RMatrix.__shared_matrix__["link2"] = link2
                RMatrix.__shared_matrix__["shape"] = shape

            initargs = (state, link1, link2, shape)

            params = zip(range(n_jobs), split[:-1], split[1:], [max_iter]*n_jobs)

            with Pool(processes=n_jobs, initializer=initializer,
                    initargs=initargs) as pool:
                pool.starmap(RMatrix.iterate, params)

            self.__state__ = RMatrix.expand_to_full_state(state_np, exclude_mask)
            self.__next_hop__ = None

        RMatrix.__shared_matrix__ = {}
        return self

    def update_runner(self, n_jobs=1, max_iter=30, record_next_hop=False):
        f_size = len(RMatrix.__idx2asn__) # full size
        p_size = f_size - len(self.__exclude__) # partial size

        exclude_mask = self.__exclude_mask__
        exclude_idx = self.__exclude_idx__

        # remap indexing
        full2partial = np.arange(f_size)
        for i in exclude_idx:
            full2partial[i:] -= 1
        full2partial[exclude_mask] = -1

        idx_ngbrs = [dict() for _ in range(p_size)]
        for i, __ngbrs__ in enumerate(RMatrix.__idx_ngbrs__):
            i = full2partial[i]
            if i == -1: continue
            for rel, __ngbr_list__ in __ngbrs__.items():
                idx_ngbrs[i][rel] = np.setdiff1d(full2partial[__ngbr_list__], [-1])
        print(f"remap indexing done")

        # init matrix
        shape = (p_size, p_size)

        state = RawArray(c_ubyte, shape[0]*shape[1]) # shared
        state_np = np.frombuffer(state, dtype=np.uint8).reshape(shape, order="F")
        state_np[:] = 0b00_111111
        for i, ngbrs in enumerate(idx_ngbrs):
            state_np[ngbrs[RMatrix.C2P], i] = 0b11_111110
            state_np[ngbrs[RMatrix.P2P], i] = 0b10_111110
            state_np[ngbrs[RMatrix.P2C], i] = 0b01_111110
        state_np[np.arange(p_size), np.arange(p_size)] = 0b11_111111
        print(f"state matrix constructed")

        link1 = RawArray(c_ubyte, shape[0]*shape[1]) # shared
        link1_np = np.frombuffer(link1, dtype=np.uint8).reshape(shape, order="C")
        for i, ngbrs in enumerate(idx_ngbrs):
            link1_np[i, ngbrs[RMatrix.P2C]] = 0b11_000000
            link1_np[i, ngbrs[RMatrix.P2P]] = 0b10_000000
            link1_np[i, ngbrs[RMatrix.C2P]] = 0b01_000000
        print(f"link1 matrix constructed")

        link2 = RawArray(c_ubyte, shape[0]*shape[1]) # shared
        link2_np = np.frombuffer(link2, dtype=np.uint8).reshape(shape, order="C")
        link2_np[:] = 0b00_111111
        for i, ngbrs in enumerate(idx_ngbrs):
            link2_np[i, ngbrs[RMatrix.C2P]] = 0b01_111111
        print(f"link2 matrix constructed")

        if record_next_hop:
            next_hop = RawArray(c_int, shape[0]*shape[1])
            next_hop_np = np.frombuffer(next_hop, dtype=np.int32).reshape(shape, order="F")
            next_hop_np[:] = -1
            print(f"next_hop matrix constructed")

        # split for parallel tasks
        assert n_jobs >= 1
        split = np.linspace(0, p_size, n_jobs+1).astype(int)
        print(f"runner with {n_jobs} processes.")

        if record_next_hop:
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
                    pool.starmap(RMatrix.iterate2, params)
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
                    pool.starmap(RMatrix.iterate, params)
        return runner

    def get_state(self, asn1, asn2):
        '''Return `(s_type, s_len)`
           s_type: One of the values from `None`, `C2P`, `P2P` and `P2C`.
                Return `P2C` if the queried `asn1` and `asn2` are the same.
           s_len: The length of the path if it exists, i.e., when `s_type`
                is not `None`. Otherwise, the meaning of `s_len` is undefined.
        '''
        if not (self.has_asn(asn1) and self.has_asn(asn2)):
            return None, 0b00_111111

        def get_state_in_matrix(asn1, asn2):
            s = self.__state__[self.asn2idx(asn1), self.asn2idx(asn2)]
            s_type = [None, RMatrix.C2P, RMatrix.P2P, RMatrix.P2C][s >> 6]
            s_len = 0b00_111111 - (s & 0b00_111111)
            return s_type, s_len

        if asn1 in self.__branch2gateway__:
            gw1, dist1, bid1, _ = self.__branch2gateway__[asn1]
            if asn2 in self.__branch2gateway__:
                gw2, dist2, bid2, _ = self.__branch2gateway__[asn2]
                if bid1 == bid2: # in the same branch
                    s_type = RMatrix.C2P if dist1 > dist2 else RMatrix.P2C
                    s_len = abs(dist1-dist2)
                # must pass through gw1/gw2 then
                elif gw1 in self.__exclude__ or gw2 in self.__exclude__:
                    s_type, s_len = None, 0b00_111111
                else:
                    s_type, s_len = get_state_in_matrix(gw1, gw2)
                    if s_type is not None:
                        s_type = RMatrix.C2P
                        s_len += dist1
                        s_len += dist2
            # must pass through gw1 then
            elif gw1 in self.__exclude__:
                s_type, s_len = None, 0b00_111111
            else:
                s_type, s_len = get_state_in_matrix(gw1, asn2)
                if s_type is not None:
                    s_type = RMatrix.C2P
                    s_len += dist1
        else:
            if asn2 in self.__branch2gateway__:
                gw2, dist2, bid2, _ = self.__branch2gateway__[asn2]
                # must pass through gw2 then
                if gw2 in self.__exclude__:
                    s_type, s_len = None, 0b00_111111
                else:
                    s_type, s_len = get_state_in_matrix(asn1, gw2)
                    if s_type is not None:
                        s_len += dist2
            else:
                s_type, s_len = get_state_in_matrix(asn1, asn2)

        return s_type, s_len

    def get_path(self, asn1, asn2):
        '''Return `path`
           path: A list of ASNs that form the AS-level path (i.e., AS_path)
               from `asn1` to `asn2`. `asn1` is not included, while `asn2`
               is always the tail of the list, if the path exists. Return
               `None` is the path doesn't exist. Return `[]` if the queired
               `asn1` and `asn2` are the same.
        '''
        assert self.__next_hop__ is not None

        if not (self.has_asn(asn1) and self.has_asn(asn2)):
            return None

        def get_path_in_matrix(asn1, asn2):
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

        if asn1 in self.__branch2gateway__:
            gw1, dist1, bid1, _ = self.__branch2gateway__[asn1]
            if asn2 in self.__branch2gateway__:
                gw2, dist2, bid2, _ = self.__branch2gateway__[asn2]
                if bid1 == bid2: # in the same branch
                    if dist1 > dist2:
                        path = []
                        ups = asn1
                        while ups != asn2:
                            _, _, _, ups = self.__branch2gateway__[ups]
                            path.append(ups)
                    else:
                        path = []
                        ups = asn2
                        while ups != asn1:
                            path.append(ups)
                            _, _, _, ups = self.__branch2gateway__[ups]
                        path = path[::-1]
                # must pass through gw1/gw2 then
                elif gw1 in self.__exclude__ or gw2 in self.__exclude__:
                    path = None
                else:
                    path = get_path_in_matrix(gw1, gw2)
                    if path is not None:
                        path1 = []
                        ups = asn1
                        while ups in self.__branch2gateway__:
                            _, _, _, ups = self.__branch2gateway__[ups]
                            path1.append(ups)

                        path2 = []
                        ups = asn2
                        while ups in self.__branch2gateway__:
                            path2.append(ups)
                            _, _, _, ups = self.__branch2gateway__[ups]

                        path = path1 + path + path2[::-1]
            # must pass through gw1 then
            elif gw1 in self.__exclude__:
                path = None
            else:
                path = get_path_in_matrix(gw1, asn2)
                if path is not None:
                    path1 = []
                    ups = asn1
                    while ups in self.__branch2gateway__:
                        _, _, _, ups = self.__branch2gateway__[ups]
                        path1.append(ups)
                    path = path1 + path
        else:
            if asn2 in self.__branch2gateway__:
                gw2, dist2, bid2, _ = self.__branch2gateway__[asn2]
                # must pass through gw2 then
                if gw2 in self.__exclude__:
                    path = None
                else:
                    path = get_path_in_matrix(asn1, gw2)
                    if path is not None:
                        path2 = []
                        ups = asn2
                        while ups in self.__branch2gateway__:
                            path2.append(ups)
                            _, _, _, ups = self.__branch2gateway__[ups]
                        path = path + path2[::-1]
            else:
                path = get_path_in_matrix(asn1, asn2)

        return path

    @staticmethod
    def expand_to_full_state(state, exclude_idx):
        f_size = len(RMatrix.__idx2asn__)
        f_state = np.full((f_size, f_size), 0b00_111111, dtype=np.uint8)
        f_state[exclude_idx, exclude_idx] = 0b11_111111

        include_2d_mask = np.ones((f_size, f_size), dtype=bool)
        include_2d_mask[exclude_idx] = False
        include_2d_mask[:, exclude_idx] = False

        f_state[include_2d_mask] = state.ravel()
        return f_state

    @staticmethod
    def expand_to_full_next_hop(next_hop, exclude_idx, remap=True):
        f_size = len(RMatrix.__idx2asn__)
        f_next_hop = np.full((f_size, f_size), -1, dtype=np.int32)

        if remap:
            partial2full = list(range(f_size))
            for i in sorted(exclude_idx)[::-1]:
                partial2full.pop(i)
            partial2full.append(-1) # index -1 keeps to -1
            partial2full = np.array(partial2full)
            for row in next_hop: # processing by row to reduce memory usage spike
                row[:] = partial2full[row]

        include_2d_mask = np.ones((f_size, f_size), dtype=bool)
        include_2d_mask[exclude_idx] = False
        include_2d_mask[:, exclude_idx] = False

        f_next_hop[include_2d_mask] = next_hop.ravel()
        return f_next_hop

    def dump(self, fpath):
        lz4dump(self, fpath)

    @staticmethod
    def load(fpath):
        assert RMatrix.__init_done__, "init class before load instances"
        return lz4load(fpath)
