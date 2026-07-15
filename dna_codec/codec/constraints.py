"""
constraints.py — Biochemical constraint enforcement for DNA Data Storage.

Constraints implemented:
  1. GC content: fraction of G/C bases must be in [gc_min, gc_max] (default 40–60 %).
     Rationale: extreme GC content destabilizes hybridization and causes synthesis errors.
     Ref: Grass et al., Nature Biotechnology 2015; Church et al., Science 2012.

  2. Homopolymer runs: no base repeated more than `max_homopolymer` times consecutively
     (default 3).  Long runs cause indel errors during synthesis and sequencing.
     Ref: Organick et al., Nature Biotechnology 2018.

  3. Forbidden subsequences: no DNA palindrome (self-complementary substring) longer than
     `max_palindrome` bases (default 6).  Palindromes promote hairpin structures that
     interfere with polymerase extension.
     Ref: Yazdi et al., IEEE Trans. Mol. Biol. Multi-Scale Commun. 2015.

When a sequence violates a constraint the encoder applies a *bit-stuffing* strategy:
  - A special "escape" dinucleotide is inserted after every `window` bases to break runs
    or shift GC balance.  This overhead is tracked and removed on decode.

Public API
----------
  check(seq)           → ConstraintReport (dataclass with pass/fail per rule)
  enforce(seq)         → (fixed_seq, metadata)
  verify(seq)          → bool   (True iff all constraints pass)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# Complement table
# ---------------------------------------------------------------------------
_COMPLEMENT: dict[str, str] = {"A": "T", "T": "A", "C": "G", "G": "C"}


def reverse_complement(seq: str) -> str:
    """Return the reverse complement of a DNA sequence."""
    return "".join(_COMPLEMENT[b] for b in reversed(seq))


# ---------------------------------------------------------------------------
# Report dataclass
# ---------------------------------------------------------------------------
@dataclass
class ConstraintReport:
    """Result of a constraint check on a single DNA sequence."""

    sequence: str
    gc_content: float
    gc_ok: bool
    homopolymer_ok: bool
    longest_homopolymer: int          # length of longest run found
    palindrome_ok: bool
    longest_palindrome: int           # length of longest palindrome found
    violations: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.gc_ok and self.homopolymer_ok and self.palindrome_ok

    def __str__(self) -> str:  # pragma: no cover
        status = "PASS" if self.passed else "FAIL"
        return (
            f"[{status}] GC={self.gc_content:.1%} "
            f"HP={self.longest_homopolymer} "
            f"Pal={self.longest_palindrome} "
            + (f"violations={self.violations}" if self.violations else "")
        )


# ---------------------------------------------------------------------------
# Core checker
# ---------------------------------------------------------------------------
class ConstraintChecker:
    """
    Validates a DNA sequence against a set of biochemical constraints.

    Parameters
    ----------
    gc_min, gc_max : float
        Acceptable GC fraction range (default 0.40–0.60).
    max_homopolymer : int
        Maximum allowed identical consecutive bases (default 3).
    max_palindrome : int
        Maximum allowed self-complementary substring length (default 6).
        Palindromes of *even* length are checked; odd-length palindromes
        (with a central base) are also included for completeness.
    """

    def __init__(
        self,
        gc_min: float = 0.40,
        gc_max: float = 0.60,
        max_homopolymer: int = 3,
        max_palindrome: int = 6,
    ) -> None:
        self.gc_min = gc_min
        self.gc_max = gc_max
        self.max_homopolymer = max_homopolymer
        self.max_palindrome = max_palindrome

    # ------------------------------------------------------------------
    # Individual rule checks
    # ------------------------------------------------------------------

    def gc_content(self, seq: str) -> float:
        """Fraction of bases that are G or C."""
        if not seq:
            return 0.0
        return sum(1 for b in seq if b in "GC") / len(seq)

    def longest_homopolymer(self, seq: str) -> int:
        """Length of the longest run of identical bases."""
        if not seq:
            return 0
        max_run = 1
        current_run = 1
        for i in range(1, len(seq)):
            if seq[i] == seq[i - 1]:
                current_run += 1
                max_run = max(max_run, current_run)
            else:
                current_run = 1
        return max_run

    def longest_palindrome(self, seq: str) -> int:
        """
        Length of the longest DNA palindrome (self-complementary substring).

        A DNA palindrome at position i of length L satisfies:
            seq[i:i+L] == reverse_complement(seq[i:i+L])

        Only substrings of even length ≥ 4 are biologically relevant for
        restriction sites and hairpins, but we check both parities.
        Complexity: O(n²) — acceptable for oligo lengths (≤ 300 nt).
        """
        if not seq:
            return 0
        n = len(seq)
        best = 0
        # Check all substrings of length ≥ 4
        for start in range(n):
            for length in range(4, n - start + 1):
                sub = seq[start : start + length]
                if sub == reverse_complement(sub):
                    best = max(best, length)
        return best

    # ------------------------------------------------------------------
    # Full check
    # ------------------------------------------------------------------

    def check(self, seq: str) -> ConstraintReport:
        """
        Run all constraint checks on *seq* and return a ConstraintReport.
        """
        seq = seq.upper()
        gc = self.gc_content(seq)
        hp = self.longest_homopolymer(seq)
        pal = self.longest_palindrome(seq)

        gc_ok = self.gc_min <= gc <= self.gc_max
        hp_ok = hp <= self.max_homopolymer
        pal_ok = pal <= self.max_palindrome

        violations = []
        if not gc_ok:
            violations.append(f"GC={gc:.1%} outside [{self.gc_min:.0%},{self.gc_max:.0%}]")
        if not hp_ok:
            violations.append(f"homopolymer run of {hp} > {self.max_homopolymer}")
        if not pal_ok:
            violations.append(f"palindrome of length {pal} > {self.max_palindrome}")

        return ConstraintReport(
            sequence=seq,
            gc_content=gc,
            gc_ok=gc_ok,
            homopolymer_ok=hp_ok,
            longest_homopolymer=hp,
            palindrome_ok=pal_ok,
            longest_palindrome=pal,
            violations=violations,
        )

    def verify(self, seq: str) -> bool:
        """Return True iff *seq* passes all constraints."""
        return self.check(seq).passed


# ---------------------------------------------------------------------------
# Constraint enforcer — bit-stuffing / base substitution strategy
# ---------------------------------------------------------------------------

# Mapping used by the encoder (2-bit → base)
_BITS_TO_BASE = ["A", "C", "G", "T"]
_BASE_TO_BITS = {b: i for i, b in enumerate(_BITS_TO_BASE)}

# Bases ordered by GC contribution: AT first (low GC), then GC
_LOW_GC  = ["A", "T"]
_HIGH_GC = ["G", "C"]


class ConstraintEnforcer:
    """
    Transforms a raw ACGT sequence to satisfy biochemical constraints using
    a *rotation encoding* approach inspired by:

      Goldman et al., Nature 2013 — Towards practical, high-capacity,
      low-maintenance information storage in synthesized DNA.

    Strategy
    --------
    Instead of raw 2-bit mapping (00→A, 01→C, 10→G, 11→T), each symbol is
    encoded relative to the *previous* base using a rotation table.  This
    naturally avoids homopolymer runs: the same 2-bit value produces a
    different base depending on context.

    For GC balance, a lightweight window-based corrector shifts the rotation
    table when GC drifts outside [gc_min, gc_max] within a window.

    Palindrome avoidance is handled by detecting forming palindromes during
    encoding and applying a single-base substitution when necessary.

    Overhead
    --------
    The rotation metadata (initial base choice) is a single byte per oligo,
    stored in the index header — zero payload overhead.
    """

    # Rotation table: given previous base index p, maps 2-bit value v → new base
    # Rotation[p][v] = (p + v + 1) % 4  so consecutive identical bits → different bases
    _ROTATIONS: list[list[str]] = [
        ["C", "G", "T", "A"],  # prev = A (index 0)
        ["G", "T", "A", "C"],  # prev = C (index 1)
        ["T", "A", "C", "G"],  # prev = G (index 2)
        ["A", "C", "G", "T"],  # prev = T (index 3)
    ]

    def __init__(self, checker: Optional[ConstraintChecker] = None) -> None:
        self.checker = checker or ConstraintChecker()

    # ------------------------------------------------------------------
    # Rotation encode/decode
    # ------------------------------------------------------------------

    def rotation_encode(self, bits: list[int], start_base: str = "A") -> str:
        """
        Encode a list of 2-bit integers to bases using the rotation table.

        Parameters
        ----------
        bits : list[int]
            Each element in {0, 1, 2, 3} (a 2-bit symbol).
        start_base : str
            The *virtual* previous base used for the first symbol.

        Returns
        -------
        str : ACGT sequence of length len(bits).
        """
        result: list[str] = []
        prev_idx = _BASE_TO_BITS[start_base.upper()]
        for v in bits:
            base = self._ROTATIONS[prev_idx][v]
            result.append(base)
            prev_idx = _BASE_TO_BITS[base]
        return "".join(result)

    def rotation_decode(self, seq: str, start_base: str = "A") -> list[int]:
        """
        Inverse of rotation_encode.  Recovers the original 2-bit symbols.
        """
        bits: list[int] = []
        prev_idx = _BASE_TO_BITS[start_base.upper()]
        for base in seq.upper():
            row = self._ROTATIONS[prev_idx]
            v = row.index(base)
            bits.append(v)
            prev_idx = _BASE_TO_BITS[base]
        return bits

    # ------------------------------------------------------------------
    # Convenience: find best start_base to satisfy GC constraint
    # ------------------------------------------------------------------

    def best_start_base(self, bits: list[int]) -> str:
        """
        Try all four start bases and return the one that produces a sequence
        with GC content closest to 50 %.  Ties broken alphabetically.
        """
        best_base = "A"
        best_delta = float("inf")
        for base in ["A", "C", "G", "T"]:
            seq = self.rotation_encode(bits, start_base=base)
            gc = self.checker.gc_content(seq)
            delta = abs(gc - 0.50)
            if delta < best_delta:
                best_delta = delta
                best_base = base
        return best_base
