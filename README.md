<h1 align="center">matrix-bgpsim: A fast and efficient matrix-based BGP simulator with GPU acceleration</h1>

<p align="center">
  <a href="https://github.com/yhchen-tsinghua/matrix-bgpsim/releases/latest"><img src="https://img.shields.io/github/release/yhchen-tsinghua/matrix-bgpsim.svg?maxAge=600" alt="Latest release" /></a>
  <a href="https://pypi.org/project/matrix-bgpsim/"><img src="https://img.shields.io/pypi/v/matrix-bgpsim.svg?maxAge=600" alt="PyPI version"></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.8%2B-brightgreen.svg?maxAge=2592000" alt="Python version"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-yellowgreen.svg?maxAge=2592000" alt="License"></a>
</p>

`matrix-bgpsim` is a Python package that provides a high-performance BGP routing simulator built on matrix operations. It enables full-scale AS-level route generation across the entire global Internet topology (77k+ ASes) in just a few hours, following the Gao-Rexford simulation model. It supports CPU backend with Python-native multiprocessing, and optionally GPU acceleration via PyTorch and CuPy backends.

&#9989; Use `matrix-bgpsim` when you need large-scale simulation to compute all-pairs AS-level routes.

&#10060; Do not use `matrix-bgpsim` if you only want to query a small number of routes on-demand.

## Table of Contents

-   [Features](#features)
-   [Installation](#installation)
    -   [From PyPI](#from-pypi)
    -   [From Source](#from-source)
    -   [Requirements](#requirements)
-   [Usage](#usage)
    -   [Initialization](#initialization)
    -   [Simulation](#simulation)
    -   [Query](#query)
    -   [Save and Load](#save-and-load)
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
# priority: [1|0|-1|None] for [customer|peer|provider|unavailable] route
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

## How It Works

### Simulation Criteria

### Topology Compression

### One-Byte Encoding

### Matrix Operations

## Contact

Copyright (c) 2025 [Yihao Chen](https://yhchen.cn)

Email: yh-chen21@mails.tsinghua.edu.cn

---
