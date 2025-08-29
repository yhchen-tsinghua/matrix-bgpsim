# matrix-bgpsim

**matrix-bgpsim** is a high-performance Internet-wide BGP simulator built on efficient matrix operations. It enables full-scale BGP simulations across the entire global Internet topology within hours, based on the Gao-Rexford model. It is suitable for generating a complete set of BGP routes between _any_ pair of ASes.

## Installation

To install from source:

```bash
cd matrix-bgpsim
pip install .
```

## Basic Usage

```python
from matrix_bgpsim import RMatrix

# Initialize class with CAIDA AS relationship data
RMatrix.init_class("./20250101.as-rel2.txt")

# Create a new simulation instance
rm = RMatrix()

# Run the simulation
rm.run(n_jobs=1, max_iter=20, record_next_hop=True)

# Query the AS path between AS123 and AS456
path = rm.get_path("123", "456")

# Save the simulation state
rm.dump("./rm.lz4")

# Load from a saved state
rm2 = RMatrix.load("./rm.lz4")
```

## Input Format

The simulator uses CAIDA AS relationship data as input, typically formatted as:

```
<ASN1>|<ASN2>|<relationship>
```

Where `relationship` is:

-   `-1`: provider-to-customer
-   `0`: peer-to-peer

---

