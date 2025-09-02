#!/usr/bin/env python
from pathlib import Path
from matrix_bgpsim import RMatrix
from correctness_test import check_correctness
import hashlib
import json
import time

script_dir = Path(__file__).resolve().parent

n_repeat = 1 # each test is repeated so many times
backends = ["cpu", "torch", "cupy"]
results = dict()

# load checksums
checksums = dict()
for line in open(script_dir/"md5-checksum.txt", "r"):
    checksum, filename = line.strip().split()
    checksums[filename] = checksum

# benchmark and correctness check
for as_rels in ["random-10000.20250101.as-rel2.txt", "core-10000.20250101.as-rel2.txt"]:
    as_rels = script_dir/as_rels

    # ensure test file integrity
    assert hashlib.md5(as_rels.read_bytes()).hexdigest() == checksums[as_rels.name]

    rmatrix = RMatrix(as_rels)
    result = results[as_rels.name] = dict()
    rmatrices = []

    for backend in backends:
        elapse_times = []

        for _ in range(n_repeat):
            t0 = time.perf_counter()
            rmatrix.run(n_jobs=40, max_iter=32, save_next_hop=True, backend=backend)
            t1 = time.perf_counter()
            elapse_times.append(t1-t0)

        result[backend] = elapse_times

        # save & load check
        rmatrix.dump(script_dir/f"rmatrix-{backend}-{as_rels.stem}.lz4")
        rmatrix = RMatrix.load(script_dir/f"rmatrix-{backend}-{as_rels.stem}.lz4")

        # correctness check against a baseline algorithm
        check_correctness(as_rels, rmatrix, n_sample=100)

        rmatrices.append(rmatrix)

    for i in range(len(rmatrices)): # consistency check for different backends
        for j in range(i+1, len(rmatrices)):
            assert (rmatrices[i].__state__ == rmatrices[j].__state__).all()
            assert (rmatrices[i].__next_hop__ == rmatrices[j].__next_hop__).all()

json.dump(results, script_dir/"results.json", indent=2)
