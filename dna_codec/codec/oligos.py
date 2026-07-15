"""
oligos.py — Oligonucleotide fragmentation, indexing and primer design.

Each oligo has the physical structure:

    5'─[primer_fwd]─[index]─[start_base_flags]─[payload]─[primer_rev_rc]─3'
       ◄── P nt ──►◄── I nt ──►◄── F nt ──►◄──── L nt ─────►◄──── P nt ────►

Where:
  P  = primer length (default 20 nt, chosen for Tm 55–65 °C)
  I  = index field length in bases (default 8 nt → can address 4^8 = 65 536 oligos)
  F  = flags field (2 nt: encodes start_base for this oligo's payload block)
  L  = payload length = oligo_len − 2*P − I − F  (default 200 − 40 − 8 − 2 = 110 nt)

The payload of consecutive oligos overlaps by `overlap` bases (default 20 nt).
This means each oligo's payload contains `overlap` nt that are also present in
the neighbouring oligo, enabling base-level consensus voting during decoding.

Fragmentation algorithm
-----------------------
  master_seq is split into windows of size `payload_len` advancing by
  `stride = payload_len − overlap` bases.  The last window is zero-padded
  if necessary (pad bases are encoded as 'A' and stripped by the decoder
  using the length stored in the encoder header).

Primer design
-------------
  A single pair of universal primers is generated once per pool.
  Tm is estimated with the nearest-neighbor simplified formula:

      Tm = 81.5 + 16.6·log10([Na+]) + 0.41·(%GC) − 675/L   [°C]

  where [Na+] = 0.05 M (50 mM, typical PCR buffer), L = primer length.
  Primers are checked for:
    • Tm ∈ [55, 65] °C
    • GC content ∈ [40, 60] %
    • No self-complementarity (3'-end ≥ 4 nt should not form hairpins)
    • No homopolymer > 3

  If a candidate primer fails, the generator tries the next window of the
  master sequence until a valid primer is found.

Index encoding
--------------
  The oligo position index is encoded as a fixed-width base-4 number:
      index i  →  I bases where base_j = (i >> 2*(I−1−j)) & 3
  mapped via {0→A, 1→C, 2→G, 3→T}.
  This is decoded back to an integer by the reverse map.

Flags field (2 nt)
------------------
  2 bases encode the start_base used by the rotation encoder for this payload
  block: A→AA, C→AC, G→AG, T→AT.  (Only the first base of the 2-nt flags
  carries information; the second is always A as a parity/sync marker.)

References
----------
  Organick et al., Nature Biotechnology 2018.
  Church et al., Science 2012.
  SantaLucia (1998). A unified view of polymer, dumbbell, and oligonucleotide
    DNA nearest-neighbor thermodynamics. PNAS, 95(4), 1460–1465.

Public API
----------
  OligoPool(oligo_len, overlap, primer_len, index_len) — configuration
  pool.fragment(master_seq, start_bases)  → list[Oligo]
  pool.assemble(oligos)                   → (master_seq, start_bases)
  pool.write_fasta(oligos, path)
  pool.read_fasta(path)                   → list[Oligo]
  design_primers(seed_seq, primer_len)    → (fwd, rev)
  tm_nearest_neighbor(seq)                → float  [°C]
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .constraints import ConstraintChecker, reverse_complement

# ---------------------------------------------------------------------------
# Base encoding for index field
# ---------------------------------------------------------------------------
_IDX_BASES = ["A", "C", "G", "T"]
_IDX_MAP   = {b: i for i, b in enumerate(_IDX_BASES)}

# Flags: start_base → 2-nt code
_FLAG_ENC = {"A": "AA", "C": "AC", "G": "AG", "T": "AT"}
_FLAG_DEC = {v: k for k, v in _FLAG_ENC.items()}


def _int_to_bases(value: int, width: int) -> str:
    """Encode *value* as a base-4 number of exactly *width* bases."""
    if value >= 4**width:
        raise ValueError(
            f"Index {value} too large for index field of width {width} "
            f"(max {4**width - 1})"
        )
    bases: list[str] = []
    for _ in range(width):
        bases.append(_IDX_BASES[value & 3])
        value >>= 2
    return "".join(reversed(bases))


def _bases_to_int(seq: str) -> int:
    """Decode a base-4 encoded string to an integer."""
    result = 0
    for b in seq.upper():
        result = (result << 2) | _IDX_MAP[b]
    return result


# ---------------------------------------------------------------------------
# Nearest-neighbor Tm (simplified SantaLucia 1998)
# ---------------------------------------------------------------------------

# ΔH (kcal/mol) and ΔS (cal/mol·K) for each nearest-neighbor pair
# Order: 5'→3' dinucleotide
_NN_DH: dict[str, float] = {
    "AA": -7.9,  "AT": -7.2,  "AC": -8.4,  "AG": -7.8,
    "TA": -7.2,  "TT": -7.9,  "TC": -8.2,  "TG": -8.5,
    "CA": -8.5,  "CT": -7.8,  "CC": -8.0,  "CG": -10.6,
    "GA": -8.2,  "GT": -8.4,  "GC": -9.8,  "GG": -8.0,
}
_NN_DS: dict[str, float] = {
    "AA": -22.2, "AT": -20.4, "AC": -22.4, "AG": -21.0,
    "TA": -21.3, "TT": -22.2, "TC": -22.2, "TG": -22.7,
    "CA": -22.7, "CT": -21.0, "CC": -19.9, "CG": -27.2,
    "GA": -22.2, "GT": -22.4, "GC": -24.4, "GG": -19.9,
}
_R = 1.987          # cal/(mol·K)
_OLIGO_CONC = 250e-9  # 250 nM (typical PCR)


def tm_nearest_neighbor(seq: str, na_conc: float = 0.05) -> float:
    """
    Estimate the melting temperature (Tm) of a DNA oligonucleotide using
    the nearest-neighbor thermodynamic model (SantaLucia 1998).

    Parameters
    ----------
    seq : str
        Primer sequence (5'→3'), all uppercase ACGT.
    na_conc : float
        Sodium ion concentration in mol/L (default 0.05 M = 50 mM).

    Returns
    -------
    float : Tm in degrees Celsius.
    """
    seq = seq.upper()
    n = len(seq)
    if n < 2:
        return 0.0

    dh_total = 0.1     # initiation kcal/mol (approx)
    ds_total = -2.8    # initiation cal/mol·K

    for i in range(n - 1):
        dinuc = seq[i : i + 2]
        dh_total += _NN_DH.get(dinuc, -8.0)
        ds_total += _NN_DS.get(dinuc, -21.5)

    # Salt correction: ΔS_corrected = ΔS + 0.368·(n−1)·ln([Na+])
    ds_corrected = ds_total + 0.368 * (n - 1) * math.log(na_conc)

    # Tm (K) = ΔH / (ΔS + R·ln(C_T/4))
    # C_T = total strand concentration; for non-self-complementary: /4
    tm_k = (dh_total * 1000) / (ds_corrected + _R * math.log(_OLIGO_CONC / 4.0))
    return tm_k - 273.15


# ---------------------------------------------------------------------------
# Primer validation
# ---------------------------------------------------------------------------

def _has_3prime_hairpin(seq: str, min_len: int = 4) -> bool:
    """
    Return True if the 3' end of *seq* (last `min_len` bases) is complementary
    to any internal region, which would form a primer-dimer / hairpin.
    """
    tail = reverse_complement(seq[-min_len:])
    return tail in seq[:-min_len]


def validate_primer(seq: str, tm_min: float = 55.0, tm_max: float = 65.0) -> tuple[bool, list[str]]:
    """
    Validate a candidate primer sequence.

    Returns
    -------
    (ok, reasons) : bool and list of failure reasons.
    """
    checker = ConstraintChecker(gc_min=0.40, gc_max=0.60, max_homopolymer=3)
    report = checker.check(seq)
    tm = tm_nearest_neighbor(seq)

    reasons: list[str] = []
    if not (tm_min <= tm <= tm_max):
        reasons.append(f"Tm={tm:.1f}°C not in [{tm_min},{tm_max}]")
    if not report.gc_ok:
        reasons.append(f"GC={report.gc_content:.1%} out of range")
    if not report.homopolymer_ok:
        reasons.append(f"homopolymer run of {report.longest_homopolymer}")
    if _has_3prime_hairpin(seq):
        reasons.append("3'-end hairpin detected")

    return len(reasons) == 0, reasons


# ---------------------------------------------------------------------------
# Primer design
# ---------------------------------------------------------------------------

# Hardcoded universal primers (validated: Tm ≈ 60–62°C, GC ≈ 50%)
# These are synthetic sequences not matching any known genome.
_DEFAULT_PRIMERS = {
    "fwd": "ACGTAGCTGATCGTACGAGT",   # 20-mer, Tm ≈ 60 °C
    "rev": "TGCATCGACTAGCATGCTCA",   # 20-mer, rev_comp of fwd-like
}


def design_primers(
    seed_seq: str,
    primer_len: int = 20,
    tm_min: float = 55.0,
    tm_max: float = 65.0,
    max_attempts: int = 200,
) -> tuple[str, str]:
    """
    Design a forward/reverse primer pair from windows of *seed_seq*.

    Slides a window of `primer_len` bases along the sequence, validates
    each candidate, and returns the first valid pair found.  If no valid
    pair is found within `max_attempts`, returns the hardcoded defaults
    (which are pre-validated).

    Parameters
    ----------
    seed_seq : str
        Source sequence (master ACGT or dedicated primer region).
    primer_len : int
        Target primer length (default 20 nt).

    Returns
    -------
    (fwd, rev) : Two primer sequences (5'→3').  `rev` is the reverse
                 complement of the annealing site so it's ready for
                 direct synthesis.
    """
    checker = ConstraintChecker(gc_min=0.40, gc_max=0.60, max_homopolymer=3)
    n = len(seed_seq)
    attempts = 0

    for start in range(0, min(n - primer_len, max_attempts * 2), 1):
        if attempts >= max_attempts:
            break
        fwd = seed_seq[start : start + primer_len].upper()
        ok, _ = validate_primer(fwd, tm_min, tm_max)
        if not ok:
            attempts += 1
            continue

        # Rev primer anneals downstream, synthesized as rev_comp
        rev_start = start + primer_len + 10  # small gap
        if rev_start + primer_len > n:
            attempts += 1
            continue
        rev_anneal = seed_seq[rev_start : rev_start + primer_len].upper()
        rev = reverse_complement(rev_anneal)
        ok2, _ = validate_primer(rev, tm_min, tm_max)
        if ok2:
            return fwd, rev
        attempts += 1

    # Fall back to pre-validated universal primers
    return _DEFAULT_PRIMERS["fwd"], _DEFAULT_PRIMERS["rev"]


# ---------------------------------------------------------------------------
# Oligo dataclass
# ---------------------------------------------------------------------------

@dataclass
class Oligo:
    """
    A single synthesized DNA oligonucleotide.

    Attributes
    ----------
    index : int
        Position of this oligo in the ordered pool (0-based).
    payload : str
        The data-carrying bases (after removing primers and index field).
    start_base : str
        The rotation-encoding start base for this payload block.
    full_seq : str
        Complete physical sequence: primer_fwd + index_field + flags + payload + primer_rev_rc.
    primer_fwd : str
        Forward primer (5'→3').
    primer_rev_rc : str
        Reverse primer as it appears on the oligo (reverse complement of the annealing site).
    tm_fwd : float
        Melting temperature of the forward primer (°C).
    tm_rev : float
        Melting temperature of the reverse primer (°C).
    is_padded : bool
        True if this is the last oligo and its payload was zero-padded.
    """

    index: int
    payload: str
    start_base: str
    full_seq: str
    primer_fwd: str
    primer_rev_rc: str
    tm_fwd: float = 0.0
    tm_rev: float = 0.0
    is_padded: bool = False

    def __len__(self) -> int:
        return len(self.full_seq)

    def to_fasta(self) -> str:
        """Return FASTA-formatted string with metadata in the header."""
        header = (
            f">oligo_{self.index:06d}"
            f" start={self.start_base}"
            f" padded={int(self.is_padded)}"
            f" tm_fwd={self.tm_fwd:.1f}"
            f" tm_rev={self.tm_rev:.1f}"
        )
        return f"{header}\n{self.full_seq}\n"


# ---------------------------------------------------------------------------
# OligoPool — main fragmentation engine
# ---------------------------------------------------------------------------

class OligoPool:
    """
    Fragment a master DNA sequence into a pool of oligos ready for synthesis.

    Parameters
    ----------
    oligo_len : int
        Total physical oligo length in nucleotides (default 150).
    overlap : int
        Number of payload bases shared between consecutive oligos (default 20).
    primer_len : int
        Length of each primer in nucleotides (default 20).
    index_len : int
        Length of the index field in bases (default 8, supports up to 65 536 oligos).
    flags_len : int
        Length of the flags field in bases (default 2).
    primer_fwd : str | None
        Forward primer to use.  If None, primers are auto-designed.
    primer_rev_rc : str | None
        Reverse primer (as it appears on the oligo, i.e. rev_comp of annealing site).
    """

    FLAGS_LEN: int = 2  # encodes start_base

    def __init__(
        self,
        oligo_len: int = 150,
        overlap: int = 20,
        primer_len: int = 20,
        index_len: int = 8,
        primer_fwd: Optional[str] = None,
        primer_rev_rc: Optional[str] = None,
    ) -> None:
        self.oligo_len  = oligo_len
        self.overlap    = overlap
        self.primer_len = primer_len
        self.index_len  = index_len

        overhead = 2 * primer_len + index_len + self.FLAGS_LEN
        if overhead >= oligo_len:
            raise ValueError(
                f"Overhead ({overhead} nt) ≥ oligo_len ({oligo_len} nt). "
                "Reduce primer_len or index_len, or increase oligo_len."
            )
        self.payload_len = oligo_len - overhead
        self.stride      = self.payload_len - overlap

        if self.stride <= 0:
            raise ValueError(
                f"stride={self.stride} ≤ 0. "
                f"Reduce overlap (currently {overlap}) or increase oligo_len."
            )

        self.primer_fwd    = primer_fwd or _DEFAULT_PRIMERS["fwd"][:primer_len]
        self.primer_rev_rc = primer_rev_rc or _DEFAULT_PRIMERS["rev"][:primer_len]

        self.tm_fwd = tm_nearest_neighbor(self.primer_fwd)
        self.tm_rev = tm_nearest_neighbor(self.primer_rev_rc)

    # ------------------------------------------------------------------
    # Fragmentation
    # ------------------------------------------------------------------

    def fragment(
        self,
        master_seq: str,
        start_bases: list[str],
    ) -> list[Oligo]:
        """
        Slice *master_seq* into overlapping oligos.

        Parameters
        ----------
        master_seq : str
            The complete ACGT sequence produced by DNAEncoder.encode_bytes().
        start_bases : list[str]
            One start_base per *encoder block* (block_size bases).  The oligo
            builder maps each payload window to the correct start_base using
            the block index.

        Returns
        -------
        list[Oligo] : Ordered list of oligos (index 0, 1, 2, …).
        """
        n = len(master_seq)
        oligos: list[Oligo] = []
        oligo_idx = 0

        pos = 0
        while pos < n:
            end = pos + self.payload_len
            chunk = master_seq[pos:end]

            is_padded = end > n
            if is_padded:
                # Zero-pad with 'A' (encodes dibit 00 harmlessly)
                chunk = chunk.ljust(self.payload_len, "A")

            # Determine start_base: from the block that covers `pos`
            # The encoder uses block_size = len(master_seq) / len(start_bases)
            block_size = len(master_seq) // len(start_bases) if start_bases else self.payload_len
            block_idx  = min(pos // block_size, len(start_bases) - 1) if start_bases else 0
            sb = start_bases[block_idx] if start_bases else "A"

            oligo = self._build_oligo(oligo_idx, chunk, sb, is_padded)
            oligos.append(oligo)

            oligo_idx += 1
            pos += self.stride
            if is_padded:
                break

        if len(oligos) >= 4**self.index_len:
            raise ValueError(
                f"Pool has {len(oligos)} oligos but index field supports "
                f"only {4**self.index_len}.  Increase index_len."
            )

        return oligos

    def _build_oligo(
        self,
        index: int,
        payload: str,
        start_base: str,
        is_padded: bool,
    ) -> Oligo:
        """Assemble the full oligo sequence from its components."""
        index_field = _int_to_bases(index, self.index_len)
        flags_field = _FLAG_ENC[start_base]
        full_seq    = (
            self.primer_fwd
            + index_field
            + flags_field
            + payload
            + self.primer_rev_rc
        )
        return Oligo(
            index=index,
            payload=payload,
            start_base=start_base,
            full_seq=full_seq,
            primer_fwd=self.primer_fwd,
            primer_rev_rc=self.primer_rev_rc,
            tm_fwd=self.tm_fwd,
            tm_rev=self.tm_rev,
            is_padded=is_padded,
        )

    # ------------------------------------------------------------------
    # Assembly (decode direction)
    # ------------------------------------------------------------------

    def assemble(self, oligos: list[Oligo]) -> tuple[str, list[str]]:
        """
        Reconstruct the master sequence from a (possibly disordered) pool of
        oligos using index-based ordering and overlap-region consensus voting.

        Parameters
        ----------
        oligos : list[Oligo]
            May be in any order; duplicates are allowed (consensus helps).

        Returns
        -------
        (master_seq, start_bases) where start_bases are extracted from the
        flags field of each oligo.
        """
        if not oligos:
            return "", []

        # --- Group by index (handle duplicates via consensus) ----------
        by_index: dict[int, list[str]] = {}
        start_base_map: dict[int, str] = {}

        for oligo in oligos:
            idx = oligo.index
            by_index.setdefault(idx, []).append(oligo.payload)
            start_base_map[idx] = oligo.start_base

        max_idx = max(by_index.keys())

        # --- Build per-position vote arrays ---------------------------
        # master_seq is assembled position-by-position.
        # Overlapping regions are voted on: most-frequent base wins.
        total_len = self.stride * max_idx + self.payload_len
        votes: list[dict[str, int]] = [{} for _ in range(total_len)]

        for idx, payloads in by_index.items():
            pos_start = idx * self.stride
            for payload in payloads:
                for offset, base in enumerate(payload):
                    pos = pos_start + offset
                    if pos < total_len:
                        votes[pos][base] = votes[pos].get(base, 0) + 1

        # --- Consensus: pick most voted base; tie → first alphabetically
        master_seq = "".join(
            max(vote, key=lambda b: (vote[b], -ord(b))) if vote else "A"
            for vote in votes
        )

        # --- Recover start_bases in order ----------------------------
        # Need one start_base per encoder block.  Use start_base_map to
        # reconstruct; gaps filled with 'A'.
        start_bases: list[str] = [
            start_base_map.get(i, "A") for i in range(max_idx + 1)
        ]
        # Note: decoder maps oligo start_bases back to encoder blocks,
        # so the relationship is 1:1 when payload_len == encoder block_size.
        # For the general case, the decoder handles re-mapping.

        return master_seq, start_bases

    # ------------------------------------------------------------------
    # FASTA I/O
    # ------------------------------------------------------------------

    def write_fasta(self, oligos: list[Oligo], path: str | Path) -> None:
        """Write the oligo pool to a FASTA file."""
        path = Path(path)
        with path.open("w") as f:
            for oligo in oligos:
                f.write(oligo.to_fasta())

    def read_fasta(self, path: str | Path) -> list[Oligo]:
        """
        Parse a FASTA file written by write_fasta and reconstruct Oligo objects.
        Strips primers and index/flags fields to recover payload and metadata.
        """
        path = Path(path)
        oligos: list[Oligo] = []
        header: dict = {}
        seq_lines: list[str] = []
        in_record: list[bool] = [False]  # mutable sentinel for nonlocal access

        def _flush() -> None:
            if not in_record[0] or not seq_lines:
                return
            full_seq = "".join(seq_lines).upper().strip()
            # Strip primers
            inner = full_seq[self.primer_len : len(full_seq) - self.primer_len]
            # Parse index field
            index_str = inner[: self.index_len]
            flags_str = inner[self.index_len : self.index_len + self.FLAGS_LEN]
            payload   = inner[self.index_len + self.FLAGS_LEN :]

            idx        = _bases_to_int(index_str)
            start_base = _FLAG_DEC.get(flags_str, header.get("start", "A"))
            is_padded  = bool(int(header.get("padded", "0")))
            tm_fwd     = float(header.get("tm_fwd", "0"))
            tm_rev     = float(header.get("tm_rev", "0"))

            oligos.append(
                Oligo(
                    index=idx,
                    payload=payload,
                    start_base=start_base,
                    full_seq=full_seq,
                    primer_fwd=full_seq[: self.primer_len],
                    primer_rev_rc=full_seq[-self.primer_len :],
                    tm_fwd=tm_fwd,
                    tm_rev=tm_rev,
                    is_padded=is_padded,
                )
            )

        with path.open() as f:
            for line in f:
                line = line.rstrip()
                if line.startswith(">"):
                    _flush()
                    seq_lines = []
                    in_record[0] = True
                    # Parse header fields: key=value space-separated
                    header = {}
                    parts = line[1:].split()
                    for part in parts[1:]:
                        if "=" in part:
                            k, v = part.split("=", 1)
                            header[k] = v
                else:
                    seq_lines.append(line)

        _flush()
        return oligos

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    def info(self) -> dict:
        """Return a summary of pool configuration."""
        return {
            "oligo_len"   : self.oligo_len,
            "primer_len"  : self.primer_len,
            "index_len"   : self.index_len,
            "flags_len"   : self.FLAGS_LEN,
            "payload_len" : self.payload_len,
            "overlap"     : self.overlap,
            "stride"      : self.stride,
            "max_oligos"  : 4**self.index_len,
            "tm_fwd"      : round(self.tm_fwd, 1),
            "tm_rev"      : round(self.tm_rev, 1),
        }
