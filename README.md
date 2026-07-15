# dna_codec

Binary-to-DNA codec with Reed-Solomon error correction. Encodes arbitrary bytes into a biochemically valid ACGT sequence, fragments it into synthesis-length oligonucleotides, and recovers the original data from a noisy, reordered pool with SHA-256 integrity verification.

## Installation

```bash
pip install -r requirements.txt
```

Runtime dependency: [`reedsolo`](https://pypi.org/project/reedsolo/) (pure-Python RS over GF(2^8)).



## Pipeline

### Stage 1

Input bytes are mapped to 2-bit symbols and rotation-encoded into ACGT bases. The rotation table is chosen per 200-nt block to maintain GC content in [40–60%] and avoid homopolymer runs (>3) and palindromes (>6 nt). A SHA-256 digest and 8-byte header are prepended to the payload for integrity verification on decode.

### Stage 2

The master sequence is split into overlapping oligos of fixed length (default 150 nt):

```
5'─[primer_fwd 20nt]─[index 8nt]─[flags 2nt]─[payload 110nt]─[primer_rev_rc 20nt]─3'
```

The index field (base-4 encoded, supports up to 65 536 oligos) allows reordering-robust reassembly. Consecutive oligos overlap by 20 nt, enabling majority-vote consensus at the base level.

### Stage 3

Two-dimensional Reed-Solomon protection:

- **Row parity** — per-oligo RS over GF(2^8); corrects up to `nsym/2` byte errors per oligo.
- **Column parity** — cross-oligo RS; extra parity oligos appended to the pool reconstruct entirely dropped oligos (analogous to RAID-6). The pool is striped into groups of at most `g` oligos where `g + nsym ≤ 255`, satisfying the GF(2^8) codeword limit.

### Stage 4

1. Approximate primer alignment (Levenshtein, `max_edit=4`) locates the inner region of each received oligo.
2. Index and start-base fields are parsed; corrupted indices are flagged and cross-validated against payload overlap.
3. Missing oligos are reconstructed via column-parity RS before per-oligo RS decode.
4. The master sequence is assembled by overlap consensus voting and decoded via `DNAEncoder.decode_sequence`; the embedded SHA-256 digest is verified.

## References

- Church, Gao, Kosuri (2012). *Science* 337(6102).
- Grass et al. (2015). *Nature Biotechnology* 33(9).
- Organick et al. (2018). *Nature Biotechnology* 36(3).
- Reed & Solomon (1960). *SIAM J. Applied Mathematics* 8(2).
