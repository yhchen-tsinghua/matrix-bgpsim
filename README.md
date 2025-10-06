<h1 align="center">matrix-bgpsim: A fast and efficient matrix-based BGP simulator with GPU acceleration</h1>

<p align="center">
  <a href="https://github.com/yhchen-tsinghua/matrix-bgpsim/releases/latest"><img src="https://img.shields.io/github/release/yhchen-tsinghua/matrix-bgpsim.svg?maxAge=600" alt="Latest release" /></a>
  <a href="https://pypi.org/project/matrix-bgpsim/"><img src="https://img.shields.io/pypi/v/matrix-bgpsim.svg?maxAge=600" alt="PyPI version"></a>
  <a href="https://matrix-bgpsim.readthedocs.io/en/latest/"><img src="https://readthedocs.org/projects/matrix-bgpsim/badge/?version=latest" alt="Docs Status" /></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.8%2B-brightgreen.svg?maxAge=2592000" alt="Python version"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-yellowgreen.svg?maxAge=2592000" alt="License"></a>
</p>

`matrix-bgpsim` is a Python project that provides a high-performance BGP routing simulator built on matrix operations. It enables full-scale AS-level route generation across the entire global Internet topology (77k+ ASes) in just a few hours, following the Gao-Rexford simulation model. It supports CPU backend with Python-native multiprocessing, and optionally GPU acceleration via PyTorch or CuPy backends.

&#9989; Use `matrix-bgpsim` when you need large-scale simulation to compute all-pairs AS-level routes.

&#10060; Do not use `matrix-bgpsim` if you only want to query a small number of routes on-demand.

## Table of Contents

-   [Features](#features)
-   [Installation](#installation)
    -   [From PyPI](#from-pypi)
    -   [From Source](#from-source)
    -   [Requirements](#requirements)
-   [Usage](#usage)
    -   [Initialization](#1-initialization)
    -   [Simulation](#2-simulation)
    -   [Query](#3-query)
    -   [Save and Load](#4-save-and-load)
-   [Documentation](#documentation)
-   [How It Works](#how-it-works)
    -   [Simulation Criteria](#simulation-criteria)
    -   [Topology Compression](#topology-compression)
    -   [One-Byte Encoding](#one-byte-encoding)
    -   [Matrix Operations](#matrix-operations)
-   [Contact](#contact)

## Features

- Full-scale simulation for all AS pairs in a single run

- High performance with matrix-based computations

- GPU acceleration with PyTorch and CuPy backends

- Compact state representation that uses one byte per route

- Flexible multiprocessing backends with CPU and GPU

- Research friendly with CAIDA-format input and Gao-Rexford model

## Installation

### From PyPI

You can install `matrix-bgpsim` with desired backends directly from PyPI:

* To install CPU-only backend (without GPU acceleration):

    ```bash
    pip install matrix-bgpsim
    ```

* To install optional PyTorch backend (for GPU acceleration with PyTorch):

    ```bash
    pip install matrix-bgpsim[torch]
    ```

* To install optional CuPy backend (for GPU acceleration with CuPy):

    ```bash
    pip install matrix-bgpsim[cupy]
    ```

* To install multiple optional backends at once:

    ```bash
    pip install matrix-bgpsim[torch,cupy]
    ```

### From Source

You can clone this repository and install `matrix-bgpsim` from source:

* To install CPU-only backend (without GPU acceleration):

    ```bash
    git clone https://github.com/yhchen-tsinghua/matrix-bgpsim.git
    cd matrix-bgpsim
    pip install .
    ```

* To install optional PyTorch backend (for GPU acceleration with PyTorch):

    ```bash
    git clone https://github.com/yhchen-tsinghua/matrix-bgpsim.git
    cd matrix-bgpsim
    pip install ".[torch]"
    ```

* To install optional CuPy backend (for GPU acceleration with CuPy):

    ```bash
    git clone https://github.com/yhchen-tsinghua/matrix-bgpsim.git
    cd matrix-bgpsim
    pip install ".[cupy]"
    ```

* To install multiple optional backends at once:

    ```bash
    git clone https://github.com/yhchen-tsinghua/matrix-bgpsim.git
    cd matrix-bgpsim
    pip install ".[torch,cupy]"
    ```

### Requirements

* Python 3.8 or higher
* `lz4` for file compression
* `numpy>=1.21` for CPU-only backend
* `torch>=2.0.0` for PyTorch backend
* `cupy>=12.0.0` for CuPy backend

## Usage

### 1. Initialization

To initialize with CAIDA-format AS relationship data:

```python
from matrix_bgpsim import RMatrix
rmatrix = RMatrix(
    input_rels="20250101.as-rel2.txt", # file path to AS relationship data
    excluded={"1234", "5678"},         # ASes to exclude from the topology
)
```

### 2. Simulation

To run simulation with a certain backend:

-   Using CPU backend:

    ```python
    rmatrix.run(
        n_jobs=4,           # number of parallel CPU processes
        max_iter=32,        # maximum route propagation iterations
        save_next_hop=True, # store next-hop information
        backend="cpu",      # use CPU multiprocessing
    )
    ```

-   Using PyTorch backend:

    ```python
    rmatrix.run(
        max_iter=32,        # maximum route propagation iterations
        save_next_hop=True, # store next-hop information
        backend="torch",    # use PyTorch CPU/GPU backend
        device="cuda:0"     # specify CPU/GPU device (defaults to "cuda:0" if available)
    )
    ```

-   Using CuPy backend:

    ```python
    rmatrix.run(
        max_iter=32,        # maximum route propagation iterations
        save_next_hop=True, # store next-hop information
        backend="cupy",     # use CuPy GPU backend
        device=0            # specify GPU ID (defaults to 0 if available)
    )
    ```

### 3. Query

To query the priority (i.e., whether it is from customer, peer, or provider) and path length of a certain route:

```python
# Query the priority and path length of the route from AS7018 to AS3356
# priority: [1|0|-1|None] for [provider|peer|customer|unavailable] route
# length: the number of AS hops
priority, length = rmatrix.get_state("7018", "3356")
```

To query the full AS path (must run the simulation with `save_next_hop=True`):
```python
# Query the full AS path from AS7018 to AS3356
# as_path: a list of ASNs, excluding "7018" and including "3356"
as_path = rmatrix.get_path("7018", "3356")
```

### 4. Save and Load

To save the computed results to disk:

```python
rmatrix.dump("rmatrix.lz4")
```

To load the computed results from disk:

```python
rmatrix = RMatrix.load("rmatrix.lz4")
```

## Documentation

The full API reference and usage guide are available at 👉 https://matrix-bgpsim.readthedocs.io

## How It Works

### Simulation Criteria

`matrix-bgpsim` follows the standard Gao-Rexford simulation principles for AS-level routing. Route selection proceeds in three steps:

* Highest local preference: routes from customers are preferred over peers, which in turn are preferred over providers.
* Shortest AS path: among routes with equal local preference, the path with fewer AS hops is chosen.
* Random tie-breaking: if multiple routes remain equivalent, the one with smallest ASN is selected (looks random).

The valley-free constraint is also enforced: routes learned from a peer or provider are only propagated to customers, preventing "valley" paths in the network.

### Topology Compression

To improve simulation efficiency, `matrix-bgpsim` compresses the Internet topology by separating _core ASes_ from _branch ASes_.

* Core ASes are the main nodes in the network that participate directly in the matrix-based BGP routing simulation. These typically include ASes with multiple providers, multiple customers, or any peers.

* Branch ASes are typically single-homed ASes or stub networks whose routing tables can be fully derived from their upstream access AS. A branch is defined recursively as a sequence of ASes starting from a stub AS and following its single provider until reaching a core AS (the access AS) with either:
    -   more than one provider,
    -   more than one customer, or
    -   any peers.

The key intuition is that all routes to or from branch ASes pass through their access AS (called "root" in the code). Therefore, branch ASes do not need to participate in the main matrix simulation. Instead:

1. The core topology is extracted by pruning all branch ASes.
2. BGP routing simulation is performed only on the core topology.
3. Routes for branch ASes are efficiently reconstructed afterwards using their access AS’s routing table, e.g., by concatenating the branch sequence.

This approach significantly reduces simulation complexity while preserving complete routing information. In practice, it can reduce topology size by over 30% in vertices, leaving only the core ASes for matrix-based computation.

### One-Byte Encoding

Route selection in BGP can be computationally expensive. `matrix-bgpsim` speeds this up using a one-byte route priority encoding, which packs both local preference and path length into a single byte.

* Structure of the byte:
    - The two most significant bits encode local preference:
        - `11`: customer route
        - `10`: peer route
        - `01`: provider route
        - `00`: unreachable origin
    - The six least significant bits store the _bitwise complement_ of the path length (max 63 hops).

* Why design so:
    - Higher byte values indicate more preferred routes.
    - Best-route selection becomes a simple max operation over the byte array.
    - Updating the route during propagation only requires basic arithmetic, e.g., subtracting one per hop to adjust the path length field.

This encoding allows `matrix-bgpsim` to turn complex BGP iterations into highly optimized matrix operations, making large-scale AS-level simulations fast and memory-efficient.

### Matrix Operations

`matrix-bgpsim` represents the network as matrices, allowing route propagation and next-hop computation to be performed in a highly vectorized way.

* State matrix: Each element encodes the route priority (local preference and path length) from a source AS to a destination AS using the one-byte encoding.

* Next-hop matrix: Stores the index of the first AS to forward a route toward a destination. It is updated simultaneously with the state matrix.

* Initialization:
    - The diagonal of the state matrix is set to enforce self-reachability.
    - Neighbor relationships (C2P, P2P, P2C) are initialized with appropriate priorities in the state matrix.
    - Link matrices (link1 and link2) help track propagation information.

* Propagation loop:
    1. Prepare state updates: Bitwise operations separate the local preference and path length fields and update them for propagation.
    2. Compute new priorities: For each destination, the algorithm evaluates all incoming routes via neighbors and selects the maximum priority route.
    3. Update next-hop matrix: The index corresponding to the chosen route becomes the next hop for that destination.
    4. Repeat: Iteration continues until the routing state converges. 

* Path reconstruction:

    Once the next-hop matrix is computed:

    1. Start from a source AS and repeatedly look up the next-hop AS for the desired destination.
    2. Follow the chain of next hops recursively until the destination is reached.
    3. This reconstructs the full AS-level path without needing to store all intermediate route information during propagation.

* Advantages:

    - Highly vectorized using NumPy, PyTorch or CuPy backends.
    - Efficient next-hop tracking allows fast path reconstruction on demand.
    - Matrix operations reduce BGP propagation to bitwise and arithmetic operations, maximizing speed and memory efficiency.

## Contact

Copyright (c) 2025 [Yihao Chen](https://yhchen.cn)

Email: yh-chen21@mails.tsinghua.edu.cn

---
