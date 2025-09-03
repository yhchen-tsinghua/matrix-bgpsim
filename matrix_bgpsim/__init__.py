from __future__ import annotations
from typing import (
    Any, Iterator, Iterable, Union, Optional, Callable, Dict, List, Tuple, Set, NamedTuple
)
from pathlib import Path

from multiprocessing import RawArray, Pool
from ctypes import c_ubyte, c_int
from collections import defaultdict
import numpy as np
import lz4.frame
import pickle

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("matrix-bgpsim")
except PackageNotFoundError:
    __version__ = "0.0.0"


class RMatrix:
    # AS relationship notations (CAIDA norms)
    P2C: int = -1 # provider-to-customer
    P2P: int =  0 # peer-to-peer
    C2P: int = +1 # customer-to-provider

    # Named tuple mapping AS relationship types:
    # - P2P: Peer-to-peer (index 0)
    # - C2P: Customer-to-provider (index +1)
    # - P2C: Provider-to-customer (index -1)
    class RelMap(NamedTuple):
        P2P: int # accessed by either index  0 or .P2P
        C2P: int # accessed by either index +1 or .C2P
        P2C: int # accessed by either index -1 or .P2C

    # Named tuple representing how a branch AS can reach the root AS of the branch:
    # - root (str): Root AS where the branch is connected
    # - next_hop (str): Next-hop ASN towards the root
    # - length (int): Number of hops to the root AS
    # - branch_id (int): Unique identifier for the branch
    class BranchRoute(NamedTuple):
        root: str      # root AS where the branch is connected
        next_hop: str  # next-hop ASN to the root AS
        length: int    # number of hops to the root AS
        branch_id: int # to identify different branches

    # Shared memory structures for multiprocessing:
    # - state: Writable matrix, dtype uint8
    # - next_hop (optional): Writable matrix, dtype int32
    # - link1: Read-only matrix, dtype uint8
    # - link2: Read-only matrix, dtype uint8
    # - shape: shape tuple, (int, int)
    __shared_matrix__: dict[str, Union[np.ndarray, tuple[int, int]]] = {
        # "state": writable, dtype uint8
        # "next_hop": (optional) writable, dtype int32
        # "link1": read-only, dtype uint8
        # "link2": read-only, dtype uint8
        # "shape": shape tuple, (int, int)
    }

    # TODO (future): add lock for thread safety?

    @staticmethod
    def caida_reader(fpath: Union[str, Path]) -> Iterator[Tuple[str, str, int]]:
        """
        Parse a CAIDA-formatted AS relationship file and yield tuples.

        Args:
            fpath: Path to the CAIDA relationship file.

        Yields:
            Tuple[str, str, int]: asn1, asn2, and relationship type (RMatrix.P2C, P2P, C2P) for each AS pair.
        """
        with open(fpath, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith("#"):
                    continue
                asn1, asn2, rel = line.split("|")[:3]
                rel = int(rel)
                yield asn1, asn2, rel

    def __init__(
        self,
        input_rels: Union[str, Path, Iterable[Tuple[str, str, int]]],
        excluded: Optional[
            Union[Callable[[str], bool], Iterable[str], Set[str], Dict[str, Any]]
        ] = None
    ) -> None:
        """
        Initialize the RMatrix object.

        Args:
            input_rels: Path to a CAIDA file or iterable of (asn1, asn2, rel) tuples.
            excluded: ASNs to exclude. Can be:
                - None (no exclusion)
                - Iterable[str] (list, tuple, set, etc.)
                - Mapping[str, Any] (membership checked via `in`)
                - Callable[[str], bool]: returns True if ASN should be excluded
        """
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

        # TODO (future): estimate memory peak

    @staticmethod
    def construct_topology(
        input_rels: Union[str, Path, Iterable[Tuple[str, str, int]]],
        excluded: Optional[
            Union[Callable[[str], bool], Iterable[str], Set[str], Dict[str, Any]]
        ] = None
    ) -> Tuple[
        List[str],               # idx2asn: core AS index → ASN
        Dict[str, int],          # asn2idx: ASN → core index
        List[RMatrix.RelMap],    # idx2ngbrs: core AS neighbors
        Dict[str, RMatrix.BranchRoute]  # asn2brts: branch ASN → branch route
    ]:
        """
        Build the AS-level topology, returning core/branch mappings.

        Args:
            input_rels: Path to a CAIDA file or iterable of (asn1, asn2, rel) tuples.
            excluded: ASNs to exclude. Can be:
                - None (no exclusion)
                - Iterable[str] (list, tuple, set, etc.)
                - Mapping[str, Any] (membership checked via `in`)
                - Callable[[str], bool]: returns True if ASN should be excluded

        Returns:
            tuple:
                - idx2asn: List of core ASNs by index
                - asn2idx: Dict from ASN to core index
                - idx2ngbrs: Core AS neighbors as RMatrix.RelMap
                - asn2brts: Dict from branch ASN to RMatrix.BranchRoute
        """
        # TODO (future): parallel simulation for disconnected sub-topologies
        # If the topology has several disconnected areas, assign each with a
        # single simluation task, so the matrix size can be greatly reduced.

        # construct AS relationships reader
        if isinstance(input_rels, (str, Path)):
            input_rels = RMatrix.caida_reader(input_rels)
        else:
            assert isinstance(input_rels, Iterable), \
                f"input_rels should be path-like or CAIDA reader: received {type(input_rels)}"

        # construct membership checker for excluded ASes
        if excluded is None or excluded is False:
            excluded_check = lambda item: False
        elif isinstance(excluded, (Set, Dict)):
            excluded_check = lambda item: item in excluded
        elif isinstance(excluded, Iterable) and not isinstance(excluded, str):
            excluded_check = lambda item: item in set(excluded)
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
        to_prune = []
        for asn, (peers, providers, customers) in asn2ngbrs.items(): # see RelMap definition
            # search from stub AS
            if len(customers) != 0 or len(peers) != 0 or len(providers) != 1:
                continue 

            # TODO (future): optimization for ASes with only one customer/peer as its neighbor
            # Routing tables of these ASes can be directly derived from their only neighbors,
            # so they can be excluded from the core network too for simulation efficiency.

            branch = []
            while (len(customers) <= 1
                    and len(peers) == 0
                        and len(providers) == 1): # search all the way up
                branch.append(asn)

                # go to the only provider
                asn, = providers 
                peers, providers, customers = asn2ngbrs[asn]

            to_prune.append((asn, branch[-1])) # where the branch should be pruned

            for i, branch_as in enumerate(branch[::-1]):
                upstream = branch[-i] if i > 0 else asn
                asn2brts[branch_as] = RMatrix.BranchRoute(
                        root=asn, next_hop=upstream, length=i+1, branch_id=branch_id)

            # A branch route means: {branch_as} can follow {next_hop} all the way up
            # along the branch, and after {length} hops, it can finally reach a root
            # AS, which can be either a core AS or the root of a dangling sub-tree.
            branch_id += 1

        for asn1, asn2 in to_prune: # prune all branches from their root ASes
            asn2ngbrs[asn1].P2C.remove(asn2) 

        # An AS could be dangling (disconnected from the core network)
        # either by default or due to the previous branch pruning
        for asn, (peers, providers, customers) in asn2ngbrs.items():
            if not peers and not providers and not customers:
                # Add a self-pointing branch route for it so it will not be included
                # in the core network later, which improves simulation efficiency.
                assert asn not in asn2brts
                asn2brts[asn] = RMatrix.BranchRoute(
                        root=asn, next_hop=None, length=0, branch_id=None)

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
        print(f"nodes: {num_nodes:,}")
        print(f"core nodes: {num_core_nodes:,} ({num_core_nodes/num_nodes:.2%})")
        print(f"edges: {num_edges:,}")
        print(f"core edges: {num_core_edges:,} ({num_core_edges/num_edges:.2%})")

        return idx2asn, asn2idx, idx2ngbrs, asn2brts

    def asn2idx(self, asn: str) -> int:
        """Return the index of a core ASN."""
        return self.__asn2idx__[asn]

    def idx2asn(self, idx: int) -> str:
        """Return the ASN corresponding to a given core AS index."""
        return self.__idx2asn__[idx]

    def idx2ngbrs(self, idx: int) -> RMatrix.RelMap:
        """Return neighbors of a core AS (as RMatrix.RelMap)."""
        return self.__idx2ngbrs__[idx]

    def asn2brts(self, asn: str) -> RMatrix.BranchRoute:
        """Return branch route details for a branch ASN."""
        return self.__asn2brts__[asn]

    def is_core_asn(self, asn: str) -> bool:
        """Check if an ASN belongs to the core network."""
        return asn in self.__asn2idx__

    def is_branch_asn(self, asn: str) -> bool:
        """Check if an ASN belongs to a branch."""
        return asn in self.__asn2brts__

    def has_asn(self, asn: str) -> bool:
        """Check if an ASN exists in the topology (core or branch)."""
        return self.is_core_asn(asn) or self.is_branch_asn(asn)

    @staticmethod
    def __iterate_state_cpu__(worker_id: int, left: int, right: int, max_iter: int) -> None:
        """
        Worker process: Propagate routing states without next-hop info.

        Args:
            worker_id: Worker identifier for logging.
            left: Start column index for this worker.
            right: End column index for this worker.
            max_iter: Maximum number of iterations to run.
        """
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
    def __iterate_state_and_next_hop_cpu__(worker_id: int, left: int, right: int, max_iter: int) -> None:
        """
        Worker process: Propagate routing states and compute next-hop info.

        Args:
            worker_id: Worker identifier for logging.
            left: Start column index for this worker.
            right: End column index for this worker.
            max_iter: Maximum number of iterations to run.
        """
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

    def run(
        self,
        n_jobs: int = 1,
        max_iter: int = 32,
        save_next_hop: bool = True,
        backend: str = "cpu",
        device: Optional[Union[str, int]] = None
    ) -> None:
        """
        Run the routing simulation.

        Args:
            n_jobs: Number of parallel processes (CPU only).
            max_iter: Maximum number of iterations.
            save_next_hop: Whether to compute next-hop matrix.
            backend: One of {"cpu", "torch", "cupy"}.
                - "cpu": Default, uses NumPy + multiprocessing.
                - "torch": GPU/CPU acceleration via PyTorch (requires `pip install matrix-bgpsim[torch]`).
                - "cupy": GPU acceleration via CuPy (requires `pip install matrix-bgpsim[cupy]`).
            device: Device specifier for GPU backends (e.g., "cuda:0", "cpu").
                Ignored for CPU backend. Defaults:
                - torch: Uses "cuda:0" if available, else "cpu".
                - cupy: Uses first available CUDA device.
        """
        if backend == "cpu":
            runner = RMatrix.__cpu_runner__(self.__idx2ngbrs__, n_jobs, max_iter, save_next_hop)
        elif backend == "torch":
            try:
                globals()["torch"] = __import__("torch")
            except ImportError:
                raise ImportError("Torch backend requires PyTorch. Install with `pip install matrix-bgpsim[torch]`.")
            if device is None:
                device = "cuda:0" if torch.cuda.is_available() else "cpu"
            runner = RMatrix.__torch_runner__(self.__idx2ngbrs__, max_iter, save_next_hop, device)
        elif backend == "cupy":
            try:
                globals()["cp"] = __import__("cupy")
            except ImportError:
                raise ImportError("CuPy backend requires CuPy. Install with `pip install matrix-bgpsim[cupy]`.")
            if device is None:
                device = 0 # CuPy uses integer device IDs
            runner = RMatrix.__cupy_runner__(self.__idx2ngbrs__, max_iter, save_next_hop, device)
        else:
            raise ValueError(f"Unsupported backend: {backend}")

        self.__state__, self.__next_hop__ = runner()

    @staticmethod
    def __cpu_runner__(idx2ngbrs: List[RMatrix.RelMap], n_jobs: int, max_iter: int, save_next_hop: bool) -> Callable[[], Tuple[Optional[np.ndarray], Optional[np.ndarray]]]:
        """
        Prepare shared memory and return a multiprocessing runner function.

        Args:
            idx2ngbrs: Core AS neighbors.
            n_jobs: Number of parallel processes.
            max_iter: Maximum iterations for state propagation.
            save_next_hop: Whether to compute next-hop info.

        Returns:
            A runner function that launches the simulation.
        """
        # init matrix
        size = len(idx2ngbrs)
        shape = (size, size)

        state = RawArray(c_ubyte, shape[0]*shape[1]) # shared raw array
        state_np = np.frombuffer(state, dtype=np.uint8).reshape(shape, order="F") # numpy interface
        state_np[:] = 0b00_111111
        for i, ngbrs in enumerate(idx2ngbrs):
            state_np[ngbrs.C2P, i] = 0b11_111110 # one-hop route to customeer i
            state_np[ngbrs.P2P, i] = 0b10_111110 # one-hop route to peer i
            state_np[ngbrs.P2C, i] = 0b01_111110 # one-hop route to provider i
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
        else: next_hop_np = None

        # split for parallel tasks
        assert n_jobs >= 1
        split = np.linspace(0, size, n_jobs+1).astype(int)
        print(f"runner with {n_jobs} processes.")

        if save_next_hop:
            def initializer(state, link1, link2, next_hop, shape):
                RMatrix.__shared_matrix__["state"] = state
                RMatrix.__shared_matrix__["link1"] = link1
                RMatrix.__shared_matrix__["link2"] = link2
                RMatrix.__shared_matrix__["next_hop"] = next_hop
                RMatrix.__shared_matrix__["shape"] = shape
            initargs = (state, link1, link2, next_hop, shape)
            process_call = RMatrix.__iterate_state_and_next_hop_cpu__
        else:
            def initializer(state, link1, link2, shape):
                RMatrix.__shared_matrix__["state"] = state
                RMatrix.__shared_matrix__["link1"] = link1
                RMatrix.__shared_matrix__["link2"] = link2
                RMatrix.__shared_matrix__["shape"] = shape
            initargs = (state, link1, link2, shape)
            process_call = RMatrix.__iterate_state_cpu__

        params = zip(range(n_jobs), split[:-1], split[1:], [max_iter]*n_jobs)

        def runner():
            with Pool(processes=n_jobs, initializer=initializer, initargs=initargs) as pool:
                pool.starmap(process_call, params)
                RMatrix.__shared_matrix__.clear()
            return state_np, next_hop_np

        return runner

    @staticmethod
    def __torch_runner__(idx2ngbrs: List[RMatrix.RelMap], max_iter: int, save_next_hop: bool, device: str = "cuda:0") -> Callable[[], Tuple[Optional[np.ndarray], Optional[np.ndarray]]]:
        """
        Prepare a runner for simulation on PyTorch backend.

        Args:
            idx2ngbrs: Core AS neighbors.
            max_iter: Maximum iterations.
            save_next_hop: Whether to compute next-hop info.
            device: PyTorch device (e.g., "cuda:0", "cpu").

        Returns:
            A runner function that performs the simulation.
        """
        # TODO (future): auto-chunking when GPU memeory is limited
        # TODO (future): multiple-GPU support
        device = torch.device(device)
        print(f"runner with {device}")

        # pre-allocate all matrices with proper memory layout
        size = len(idx2ngbrs)
        shape = (size, size)

        state = torch.full(shape, 0b00_111111, dtype=torch.uint8,
                    device=device, requires_grad=False).t() # column-major
        link1 = torch.zeros(shape, dtype=torch.uint8, device=device, requires_grad=False)
        link2 = torch.full(shape, 0b00_111111, dtype=torch.uint8, device=device, requires_grad=False)

        tmp0 = torch.empty_like(state) # column-major: assert tmp0.stride(0) == 1
        tmp1 = torch.empty_like(state) # column-major: assert tmp1.stride(0) == 1
        tmp2 = torch.empty_like(link1)
        tmp3 = torch.empty_like(link2)

        next_hop = torch.full(shape, -1, dtype=torch.int32,
                        device=device, requires_grad=False).t() if save_next_hop else None

        # initialize state and link matrices
        with torch.no_grad():
            state.fill_diagonal_(0b11_111111)
            for i, ngbrs in enumerate(idx2ngbrs):
                state[ngbrs.C2P, i] = 0b11_111110
                state[ngbrs.P2P, i] = 0b10_111110
                state[ngbrs.P2C, i] = 0b01_111110
                link1[i, ngbrs.P2C] = 0b11_000000
                link1[i, ngbrs.P2P] = 0b10_000000
                link1[i, ngbrs.C2P] = 0b01_000000
                link2[i, ngbrs.C2P] = 0b01_111111

        # the preparation step called before updating state/next_hop
        def pre_update():
            torch.bitwise_and(state, 0b11_000000, out=tmp0)
            torch.bitwise_left_shift(tmp0, 1, out=tmp1)
            torch.bitwise_right_shift(tmp0.t(), 1, out=tmp2)
            torch.bitwise_or(tmp1, tmp2.t(), out=tmp1)
            torch.bitwise_and(tmp1, 0b11_000000, out=tmp1)
            torch.bitwise_and(tmp0, tmp1, out=tmp0)
            torch.bitwise_or(tmp1, state, out=tmp1)

        if save_next_hop:
            def iterate():
                pre_update()
                for j in range(state.size(1)):
                    torch.bitwise_and(link1, tmp0[:, j], out=tmp2)
                    torch.bitwise_and(link2, tmp1[:, j], out=tmp3)
                    torch.bitwise_or(tmp2, tmp3, out=tmp2)

                    max_vals, max_idx = torch.max(tmp2, dim=1)
                    state[:, j] = max_vals.sub_(1)
                    next_hop[:, j] = max_idx
                state.fill_diagonal_(0b11_111111)
                # TODO (future): add batch processing for this iteration,
                # and automatically estimate batch size by available memory
        else:
            def iterate():
                pre_update()
                for j in range(state.size(1)):
                    torch.bitwise_and(link1, tmp0[:, j], out=tmp2)
                    torch.bitwise_and(link2, tmp1[:, j], out=tmp3)
                    torch.bitwise_or(tmp2, tmp3, out=tmp2)
                    state[:, j] = torch.max(tmp2, dim=1)[0].sub_(1)
                state.fill_diagonal_(0b11_111111)

        def runner():
            with torch.no_grad():
                prev_hash = torch.sum(state, dtype=torch.int64)
                for it in range(max_iter):
                    iterate()
                    new_hash = torch.sum(state, dtype=torch.int64)
                    print(f"Iteration {it+1} completed")
                    if torch.equal(new_hash, prev_hash): # early stop
                        break
                    prev_hash = new_hash
                if next_hop is not None:
                    next_hop[state <= 0b00_111111] = -1
                # TODO (future): write a custom CUDA kernel to reduce the mask tensor here
                # TODO (future): fine-grained early stop mechanism
            return state.cpu().numpy(), next_hop.cpu().numpy() if next_hop is not None else None

        return runner

    @staticmethod
    def __cupy_runner__(idx2ngbrs: List[RMatrix.RelMap], max_iter: int, save_next_hop: bool, device: int = 0) -> Callable[[], Tuple[Optional[np.ndarray], Optional[np.ndarray]]]:
        """
        Prepare and execute simulation on CuPy backend.

        Args:
            idx2ngbrs: Core AS neighbors.
            max_iter: Maximum iterations.
            save_next_hop: Whether to compute next-hop info.
            device: CuPy device ID (e.g., 0 for "cuda:0").

        Returns:
            A runner function that performs the simulation.
        """
        with cp.cuda.Device(device):
            print(f"runner with device {device}")
            size = len(idx2ngbrs)
            shape = (size, size)

            # pre-allocate all matrices with proper memory layout
            state = cp.full(shape, 0b00_111111, dtype=cp.uint8).T  # column-major
            link1 = cp.zeros(shape, dtype=cp.uint8)
            link2 = cp.full(shape, 0b00_111111, dtype=cp.uint8)

            tmp0 = cp.empty_like(state)
            tmp1 = cp.empty_like(state)
            tmp2 = cp.empty_like(link1)
            tmp3 = cp.empty_like(link2)

            next_hop = cp.full(shape, -1, dtype=cp.int32).T if save_next_hop else None

            # initialize state and link matrices
            cp.fill_diagonal(state, 0b11_111111)
            for i, ngbrs in enumerate(idx2ngbrs):
                state[ngbrs.C2P, i] = 0b11_111110
                state[ngbrs.P2P, i] = 0b10_111110
                state[ngbrs.P2C, i] = 0b01_111110
                link1[i, ngbrs.P2C] = 0b11_000000
                link1[i, ngbrs.P2P] = 0b10_000000
                link1[i, ngbrs.C2P] = 0b01_000000
                link2[i, ngbrs.C2P] = 0b01_111111

            # the preparation step called before updating state/next_hop
            def pre_update():
                cp.bitwise_and(state, 0b11_000000, out=tmp0)
                cp.left_shift(tmp0, 1, out=tmp1)
                cp.right_shift(tmp0.T, 1, out=tmp2)
                cp.bitwise_or(tmp1, tmp2.T, out=tmp1)
                cp.bitwise_and(tmp1, 0b11_000000, out=tmp1)
                cp.bitwise_and(tmp0, tmp1, out=tmp0)
                cp.bitwise_or(tmp1, state, out=tmp1)

            if save_next_hop:
                def iterate():
                    pre_update()
                    for j in range(state.shape[1]):
                        cp.bitwise_and(link1, tmp0[:, j], out=tmp2)
                        cp.bitwise_and(link2, tmp1[:, j], out=tmp3)
                        cp.bitwise_or(tmp2, tmp3, out=tmp2)

                        state[:, j] = cp.max(tmp2, axis=1) - 1
                        next_hop[:, j] = cp.argmax(tmp2, axis=1)
                    cp.fill_diagonal(state, 0b11_111111)
            else:
                def iterate():
                    pre_update()
                    for j in range(state.shape[1]):
                        cp.bitwise_and(link1, tmp0[:, j], out=tmp2)
                        cp.bitwise_and(link2, tmp1[:, j], out=tmp3)
                        cp.bitwise_or(tmp2, tmp3, out=tmp2)
                        state[:, j] = cp.max(tmp2, axis=1) - 1
                    cp.fill_diagonal(state, 0b11_111111)

            def runner():
                prev_hash = cp.sum(state, dtype=cp.int64)
                for it in range(max_iter):
                    iterate()
                    new_hash = cp.sum(state, dtype=cp.int64)
                    print(f"Iteration {it+1} completed")
                    if cp.array_equal(new_hash, prev_hash):  # early stop
                        break
                    prev_hash = new_hash
                if next_hop is not None:
                    next_hop[state <= 0b00_111111] = -1
                return state.get(), next_hop.get() if next_hop is not None else None

            return runner

    def get_state(self, asn1: str, asn2: str) -> Tuple[Optional[int], int]:
        """
        Get the route priority and path length between two ASNs.

        **Important caveats:**
            - Simulation (`run()`) must have been executed before calling this method,
              otherwise an AssertionError is raised.
            - If the route from `asn1` to `asn1` does not exist, the result is `(None, 0)`.
            - If `asn1 == asn2`, the type defaults to `P2C` and length is 0.

        Args:
            asn1: Source ASN.
            asn2: Destination ASN.

        Returns:
            tuple:
                - s_type: Relationship type (P2C, P2P, C2P) or None if unreachable.
                - s_len: Path length (0 if unreachable or same ASN).
        """
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

    def get_path(self, asn1: str, asn2: str) -> Optional[List[str]]:
        """
        Retrieve the AS-level path from `asn1` to `asn2`.

        **Important caveats:**
            - Simulation (`run(save_next_hop=True)`) must have been executed before
              calling this method, otherwise an AssertionError is raised.
            - If no route exists between `asn1` and `asn2`, returns `None`.
            - If `asn1 == asn2`, returns an empty list `[]` because no hops are needed.

        Args:
            asn1 : Source ASN.
            asn2 : Destination ASN.

        Returns:
            - A list of ASNs forming the path from `asn1` to `asn2`
              (excluding `asn1`, including `asn2`).
            - `[]` if `asn1 == asn2`.
            - `None` if no path exists.
        """
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

    def dump(self, fpath: Union[str, Path]) -> None:
        """Serialize and save the current RMatrix object to a lz4 file."""
        pickle.dump(self, lz4.frame.open(fpath, "wb"), protocol=4)

    @staticmethod
    def load(fpath: Union[str, Path]) -> RMatrix:
        """Load an RMatrix object from a serialized lz4 file."""
        return pickle.load(lz4.frame.open(fpath, "rb"))
