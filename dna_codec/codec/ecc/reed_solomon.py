"""
ecc/reed_solomon.py — Reed-Solomon error correction for DNA Data Storage.

Two levels of protection (RAID-like, "2D RS"):

  Level 1 — Per-oligo RS (horizontal):
    Each oligo's payload bytes are protected by `nsym` parity bytes appended
    to the payload.  Corrects up to nsym/2 byte errors per oligo.

  Level 2 — Cross-oligo RS (vertical / "interleaved"):
    For each byte column j across all N oligos, the N values at position j
    form a codeword protected by `nsym_col` parity symbols.  The parity
    codewords are stored as additional "parity oligos" at the end of the pool.
    This recovers entire lost (dropout) oligos — analogous to RAID-6.

Mathematical background
-----------------------
  Reed-Solomon codes operate over GF(2^8) (Galois Field with 256 elements).
  A (n, k) RS code takes k data symbols and produces n−k parity symbols such
  that any n−k erasures or n−k/2 errors can be corrected.

  We use the `reedsolo` library (pure Python, no C extension required) which
  implements RS over GF(2^8) with generator polynomial primitive x^8+x^4+x^3+x^2+1
  (0x11d), consistent with QR Code and many storage standards.

  Ref: Reed, I.S. & Solomon, G. (1960). Polynomial codes over certain finite
    fields. SIAM Journal on Applied Mathematics, 8(2), 300–304.
  Ref: Wicker, S.B. & Bhargava, V.K. (1994). Reed-Solomon Codes and Their
    Applications. IEEE Press.

Public API
----------
  RSCodec(redundancy=0.30)        — construct codec
  codec.encode_oligo(payload_bytes)   → encoded_bytes (payload + parity)
  codec.decode_oligo(encoded_bytes)   → (payload_bytes, n_errors_corrected)
  codec.encode_pool(oligos)           → list[Oligo]  (payloads RS-encoded)
  codec.decode_pool(oligos)           → (list[Oligo], PoolDecodeReport)
  codec.add_column_parity(oligos)     → list[Oligo]  (pool + parity oligos)
  codec.recover_with_column_parity(oligos) → list[Oligo]
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import reedsolo

from dna_codec.codec.oligos import Oligo, OligoPool, _FLAG_ENC, _int_to_bases

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GF_EXP   = 8           # GF(2^8)
GF_PRIM  = 0x11d       # primitive polynomial x^8+x^4+x^3+x^2+1
FCRS     = 1           # first consecutive root of generator polynomial


# ---------------------------------------------------------------------------
# Report dataclass
# ---------------------------------------------------------------------------
@dataclass
class OligoDecodeResult:
    """Result of decoding a single oligo."""
    index: int
    success: bool
    n_errors: int = 0
    error_message: str = ""


@dataclass
class PoolDecodeReport:
    """Aggregated report for a full pool decode."""
    total_oligos: int
    decoded_ok: int
    failed: int
    total_errors_corrected: int
    oligo_results: list[OligoDecodeResult] = field(default_factory=list)
    column_parity_used: bool = False
    oligos_recovered_by_column: int = 0

    @property
    def success_rate(self) -> float:
        return self.decoded_ok / max(self.total_oligos, 1)

    def summary(self) -> str:
        return (
            f"RS decode: {self.decoded_ok}/{self.total_oligos} OK "
            f"({self.success_rate:.1%}), "
            f"{self.total_errors_corrected} errors corrected, "
            f"{self.failed} failed"
            + (f", {self.oligos_recovered_by_column} recovered by column parity"
               if self.column_parity_used else "")
        )


# ---------------------------------------------------------------------------
# Core RS codec
# ---------------------------------------------------------------------------
class RSCodec:
    """
    Two-level Reed-Solomon codec for an oligo pool.

    Parameters
    ----------
    redundancy : float
        Fraction of payload bytes used as RS parity (default 0.30 = 30%).
        E.g. for a 100-byte payload, nsym = 30 parity bytes, so a (130,100)
        code that corrects up to 15 byte errors per oligo.
    col_redundancy : float
        Fraction of oligos used as column-parity oligos (default 0.20).
        E.g. for 100 data oligos, 20 column-parity oligos are added.
    """

    def __init__(
        self,
        redundancy: float = 0.30,
        col_redundancy: float = 0.20,
    ) -> None:
        self.redundancy = redundancy
        self.col_redundancy = col_redundancy
        # RSCodec is stateless w.r.t. payload length; nsym is set per-call
        # because payload lengths can vary (last oligo may differ).

    def _make_rs(self, nsym: int) -> reedsolo.RSCodec:
        """Create a reedsolo.RSCodec instance for *nsym* parity symbols.
        prim=285 (0x11d) is the default GF(2^8) primitive polynomial."""
        return reedsolo.RSCodec(nsym, fcr=FCRS, prim=GF_PRIM)

    # ------------------------------------------------------------------
    # Level 1: per-oligo
    # ------------------------------------------------------------------

    def nsym_for(self, payload_len: int) -> int:
        """Number of parity bytes for a payload of *payload_len* bytes."""
        return max(2, round(payload_len * self.redundancy))

    def encode_oligo(self, payload_bytes: bytes) -> bytes:
        """
        RS-encode a single oligo payload.

        Returns payload_bytes + parity_bytes.
        The total length is len(payload_bytes) + nsym.
        """
        nsym = self.nsym_for(len(payload_bytes))
        rs = self._make_rs(nsym)
        encoded = rs.encode(payload_bytes)
        return bytes(encoded)

    def decode_oligo(
        self,
        encoded_bytes: bytes,
        payload_len: int,
        erasures: Optional[list[int]] = None,
    ) -> tuple[bytes, int]:
        """
        RS-decode a single encoded oligo.

        Parameters
        ----------
        encoded_bytes : bytes
            payload + parity (as produced by encode_oligo).
        payload_len : int
            Expected length of the original payload (needed to compute nsym).
        erasures : list[int] | None
            Byte positions known to be erased (e.g. from indel detection).
            Erasures cost half an error correction symbol each.

        Returns
        -------
        (payload_bytes, n_errors_corrected)

        Raises
        ------
        reedsolo.ReedSolomonError : if the codeword is uncorrectable.
        """
        nsym = self.nsym_for(payload_len)
        rs = self._make_rs(nsym)
        decoded, _, errata = rs.decode(
            encoded_bytes, erase_pos=erasures or []
        )
        n_errors = len(errata) if errata else 0
        return bytes(decoded), n_errors

    # ------------------------------------------------------------------
    # Level 1: pool encode/decode
    # ------------------------------------------------------------------

    def encode_pool(self, oligos: list[Oligo]) -> list[Oligo]:
        """
        Apply per-oligo RS encoding to every oligo in the pool.

        The encoded bytes are stored back as the oligo's payload in base-4
        DNA encoding (each byte → 4 bases using raw 2-bit mapping).
        The full_seq of each oligo is rebuilt to include the parity bases.

        Returns a new list of Oligo objects with extended payloads.
        """
        result = []
        for oligo in oligos:
            payload_bytes = _dna_to_bytes(oligo.payload)
            encoded_bytes = self.encode_oligo(payload_bytes)
            new_payload   = _bytes_to_dna(encoded_bytes)
            new_oligo = _replace_payload(oligo, new_payload)
            result.append(new_oligo)
        return result

    def decode_pool(
        self,
        oligos: list[Oligo],
        original_payload_bases: int,
    ) -> tuple[list[Oligo], PoolDecodeReport]:
        """
        Apply per-oligo RS decoding to every oligo in the pool.

        Parameters
        ----------
        oligos : list[Oligo]
            Pool with RS-encoded payloads (as produced by encode_pool).
        original_payload_bases : int
            Number of bases in the *original* (pre-RS) payload.  Used to
            compute nsym so we know where parity bytes start.

        Returns
        -------
        (decoded_oligos, report)
        """
        original_payload_bytes = original_payload_bases // 4
        results   = []
        decoded   = []
        n_total_errors = 0

        for oligo in oligos:
            encoded_bytes = _dna_to_bytes(oligo.payload)
            try:
                payload_bytes, n_err = self.decode_oligo(encoded_bytes, original_payload_bytes)
                new_payload = _bytes_to_dna(payload_bytes)
                new_oligo   = _replace_payload(oligo, new_payload)
                decoded.append(new_oligo)
                n_total_errors += n_err
                results.append(OligoDecodeResult(
                    index=oligo.index, success=True, n_errors=n_err
                ))
            except Exception as e:
                # Uncorrectable — keep original but mark as failed
                decoded.append(oligo)
                results.append(OligoDecodeResult(
                    index=oligo.index, success=False, error_message=str(e)
                ))

        n_ok     = sum(1 for r in results if r.success)
        n_failed = len(results) - n_ok

        report = PoolDecodeReport(
            total_oligos=len(oligos),
            decoded_ok=n_ok,
            failed=n_failed,
            total_errors_corrected=n_total_errors,
            oligo_results=results,
        )
        return decoded, report

    # ------------------------------------------------------------------
    # Level 2: column (cross-oligo) parity
    # ------------------------------------------------------------------

    def add_column_parity(
        self,
        oligos: list[Oligo],
        pool: OligoPool,
    ) -> list[Oligo]:
        """
        Add column-parity oligos to the pool (Level-2 RS).

        For each byte position j across all N data oligos, collect the column
        vector [oligo_0[j], oligo_1[j], ..., oligo_{N-1}[j]] and compute RS
        parity over it.  The parity symbols form new "column-parity oligos"
        appended to the pool.

        The number of column-parity oligos is ceil(N * col_redundancy).

        A marker byte (0xFF) in the index field distinguishes parity oligos
        from data oligos so the decoder can find them.
        """
        if not oligos:
            return []

        # Ensure all payload DNA strings are the same length (pad if needed)
        payload_len = max(len(o.payload) for o in oligos)
        cols        = payload_len // 4  # bytes per oligo

        N = len(oligos)

        # GF(2^8) codewords are limited to 255 symbols total (data + parity).
        # A single RS codeword spanning all N data oligos is only valid
        # while N + nsym <= 255; beyond that reedsolo silently produces a
        # truncated/corrupt result (or raises once nsym >= nsize), which
        # is what was causing the IndexError with 625+ oligos. Note nsym
        # itself is a fraction of the group size (col_redundancy), so for
        # large N, a *global* nsym = round(N*col_redundancy) can already
        # exceed 255 on its own, before grouping is even applied — so
        # group_size and nsym must be solved for together, not nsym first.
        #
        # Find the largest group_size g such that
        #   g + round(g * col_redundancy) <= 255
        # This keeps every group's codeword (data + its own parity) within
        # the GF(2^8) symbol limit. Each group gets independent column
        # parity, analogous to RAID-6 across striped groups rather than
        # one giant stripe.
        group_size = 1
        for candidate in range(1, 256):
            g_nsym = max(2, round(candidate * self.col_redundancy))
            if candidate + g_nsym <= 255:
                group_size = candidate
            else:
                break
        group_size = min(N, group_size) if N else 0
        nsym = max(2, round(group_size * self.col_redundancy)) if group_size else 0
        n_groups = math.ceil(N / group_size) if group_size else 0

        parity_oligos: list[Oligo] = []
        base_idx = len(oligos)  # parity oligos start after all data oligos
        ref_oligo = oligos[0]
        parity_slot = 0

        for g in range(n_groups):
            g_start = g * group_size
            g_end = min(N, g_start + group_size)
            group_oligos = oligos[g_start:g_end]
            g_N = len(group_oligos)

            # Build matrix for this group: g_N rows × cols columns (bytes)
            matrix: list[bytes] = [
                _dna_to_bytes(o.payload.ljust(payload_len, "A"))
                for o in group_oligos
            ]

            rs = self._make_rs(nsym)
            col_parities: list[list[int]] = [[] for _ in range(nsym)]

            for j in range(cols):
                col_data = bytes(
                    row[j] if j < len(row) else 0 for row in matrix
                )
                col_encoded = rs.encode(col_data)
                parity_syms = col_encoded[g_N:]     # last nsym symbols
                for k, sym in enumerate(parity_syms):
                    col_parities[k].append(sym)

            # Each of the nsym parity rows for this group becomes a new
            # "parity oligo". Slot indices are allocated sequentially
            # across all groups so recover_with_column_parity can tell
            # which group and which parity slot within it each parity
            # oligo belongs to (group = slot // nsym, slot_in_group =
            # slot % nsym).
            for k, parity_row in enumerate(col_parities):
                parity_bytes = bytes(parity_row[:cols])
                parity_payload = _bytes_to_dna(parity_bytes)
                parity_idx = base_idx + parity_slot
                p_oligo = _rebuild_oligo(
                    ref_oligo, parity_idx, parity_payload, pool, is_parity=True
                )
                parity_oligos.append(p_oligo)
                parity_slot += 1

        return oligos + parity_oligos

    def recover_with_column_parity(
        self,
        oligos: list[Oligo],
        n_data_oligos: int,
        pool: OligoPool,
        failed_indices: Optional[list[int]] = None,
    ) -> tuple[list[Oligo], int]:
        """
        Use column-parity oligos to recover lost or uncorrectable data oligos.

        Parameters
        ----------
        oligos : list[Oligo]
            Pool including parity oligos (appended by add_column_parity).
        n_data_oligos : int
            Number of data oligos (not counting parity oligos).
        pool : OligoPool
            The pool configuration (needed to rebuild recovered oligos).
        failed_indices : list[int] | None
            Indices of known-lost data oligos.  If None, any index ≥
            n_data_oligos is treated as a parity oligo automatically.

        Returns
        -------
        (recovered_oligos, n_recovered)
            recovered_oligos contains only the n_data_oligos data oligos,
            with lost ones filled in via RS column decoding.
        """
        # Separate data and parity oligos
        by_index   = {o.index: o for o in oligos}
        data_oligos = [by_index.get(i) for i in range(n_data_oligos)]
        parity_oligos = [
            o for o in oligos if o.index >= n_data_oligos
        ]
        parity_oligos.sort(key=lambda o: o.index)

        if not parity_oligos:
            # No parity oligos available — return as-is
            return [o for o in data_oligos if o is not None], 0

        payload_len = max(
            (len(o.payload) for o in oligos if o is not None), default=0
        )
        cols = payload_len // 4

        N = n_data_oligos

        # add_column_parity solves for the largest group_size g such that
        # g + round(g * col_redundancy) <= 255, then uses
        # nsym = round(group_size * col_redundancy) per group, and
        # allocates parity-oligo indices sequentially across groups:
        # group = slot // nsym, slot_in_group = slot % nsym. Recompute
        # the identical grouping here so recovery lines up with how
        # encoding actually split the pool.
        group_size = 1
        for candidate in range(1, 256):
            g_nsym = max(2, round(candidate * self.col_redundancy))
            if candidate + g_nsym <= 255:
                group_size = candidate
            else:
                break
        group_size = min(N, group_size) if N else 0
        nsym = max(2, round(group_size * self.col_redundancy)) if group_size else 0
        n_groups = math.ceil(N / group_size) if group_size else 0

        # Total parity oligos actually created = n_groups * nsym. If the
        # highest surviving parity index implies more groups than our
        # local computation (e.g. redundancy settings drifted), trust
        # the larger of the two so we don't under-allocate nsym slots.
        parity_by_global_slot = {o.index - n_data_oligos: o for o in parity_oligos}
        max_slot_seen = max(parity_by_global_slot.keys(), default=-1)
        n_groups = max(n_groups, (max_slot_seen // nsym) + 1 if nsym else 0)

        n_recovered = 0
        ref = next((o for o in data_oligos if o is not None), None)

        for g in range(n_groups):
            g_start = g * group_size
            g_end = min(N, g_start + group_size)
            if g_start >= g_end:
                continue
            group_indices = list(range(g_start, g_end))
            g_N = len(group_indices)

            # Parity oligos belonging to this group, keyed by their
            # slot-within-group (0 .. nsym-1).
            group_parity_by_slot = {
                slot - g * nsym: o
                for slot, o in parity_by_global_slot.items()
                if g * nsym <= slot < (g + 1) * nsym
            }
            missing_parity_slots = [
                k for k in range(nsym) if k not in group_parity_by_slot
            ]

            # Data erasures within this group, as indices into the
            # group's own data portion (0 .. g_N-1).
            data_erasure_positions = [
                i - g_start
                for i in group_indices
                if data_oligos[i] is None
                or (failed_indices and i in failed_indices)
            ]
            data_erasure_positions = sorted(set(data_erasure_positions))

            if not data_erasure_positions or ref is None:
                continue

            # reedsolo's erase_pos is relative to the FULL codeword (data
            # symbols first, then parity symbols), so parity erasures
            # must be offset by g_N, not left at their bare slot index.
            codeword_erasure_positions = sorted(
                data_erasure_positions + [g_N + k for k in missing_parity_slots]
            )

            if len(codeword_erasure_positions) > nsym:
                # Uncorrectable within this group — leave as None, will
                # be filled with a dummy payload below.
                continue

            rs = self._make_rs(nsym)
            recovered_cols: dict[int, list[int]] = {
                i: [] for i in data_erasure_positions
            }

            for j in range(cols):
                # Build received codeword: data + parity, in original
                # group-relative order.
                received = []
                for gi, i in enumerate(group_indices):
                    if data_oligos[i] is not None:
                        row_bytes = _dna_to_bytes(
                            data_oligos[i].payload.ljust(payload_len, "A")
                        )
                        received.append(row_bytes[j] if j < len(row_bytes) else 0)
                    else:
                        received.append(0)   # placeholder for erasure

                # Parity segment must preserve original slot order
                # (slot k == the k-th parity symbol generated for this
                # column at encode time), with 0 placeholders for any
                # parity oligo that was itself lost.
                for k in range(nsym):
                    p_oligo = group_parity_by_slot.get(k)
                    if p_oligo is not None:
                        p_bytes = _dna_to_bytes(
                            p_oligo.payload.ljust(payload_len, "A")
                        )
                        received.append(p_bytes[j] if j < len(p_bytes) else 0)
                    else:
                        received.append(0)   # placeholder for erasure

                try:
                    decoded_col, _, _ = rs.decode(
                        bytes(received), erase_pos=codeword_erasure_positions
                    )
                    for ep in data_erasure_positions:
                        if ep < len(decoded_col):
                            recovered_cols[ep].append(decoded_col[ep])
                        else:
                            recovered_cols[ep].append(0)
                except Exception:
                    for ep in data_erasure_positions:
                        recovered_cols[ep].append(0)

            # Rebuild recovered oligos (translate group-relative index
            # back to the global data-oligo index).
            for ep in data_erasure_positions:
                global_ep = g_start + ep
                rec_bytes   = bytes(recovered_cols[ep])
                rec_payload = _bytes_to_dna(rec_bytes)
                rec_oligo   = _rebuild_oligo(
                    ref, global_ep, rec_payload, pool, is_parity=False
                )
                data_oligos[global_ep] = rec_oligo
                n_recovered += 1

        # Fill any still-None slots with a zero-payload oligo
        ref = next((o for o in data_oligos if o is not None), None)
        final = []
        for i, o in enumerate(data_oligos):
            if o is not None:
                final.append(o)
            elif ref is not None:
                dummy_payload = "A" * (payload_len or pool.payload_len * 4)
                final.append(_rebuild_oligo(ref, i, dummy_payload, pool))

        return final, n_recovered


# ---------------------------------------------------------------------------
# Helpers: DNA ↔ bytes (raw 2-bit mapping, no rotation — RS operates on bytes)
# ---------------------------------------------------------------------------

_B2D = ["A", "C", "G", "T"]   # 0→A, 1→C, 2→G, 3→T
_D2B = {b: i for i, b in enumerate(_B2D)}


def _bytes_to_dna(data: bytes) -> str:
    """Convert raw bytes to a DNA string (4 bases per byte, MSB-first)."""
    bases: list[str] = []
    for byte in data:
        bases.append(_B2D[(byte >> 6) & 3])
        bases.append(_B2D[(byte >> 4) & 3])
        bases.append(_B2D[(byte >> 2) & 3])
        bases.append(_B2D[byte & 3])
    return "".join(bases)


def _dna_to_bytes(dna: str) -> bytes:
    """Convert a DNA string (4 bases per byte) to bytes. Truncates trailing bases."""
    dna = dna.upper()
    n   = (len(dna) // 4) * 4   # round down to multiple of 4
    result: list[int] = []
    for i in range(0, n, 4):
        byte = (
            (_D2B.get(dna[i],   0) << 6)
            | (_D2B.get(dna[i+1], 0) << 4)
            | (_D2B.get(dna[i+2], 0) << 2)
            |  _D2B.get(dna[i+3], 0)
        )
        result.append(byte)
    return bytes(result)


def _replace_payload(oligo: Oligo, new_payload: str) -> Oligo:
    """Return a copy of *oligo* with a different payload (and rebuilt full_seq)."""
    p = len(oligo.primer_fwd)
    # Rebuild full_seq: primer_fwd + everything_between + primer_rev_rc
    # Keep index and flags from original full_seq
    header_part = oligo.full_seq[p : len(oligo.full_seq) - p - len(oligo.payload)]
    new_full    = oligo.primer_fwd + header_part + new_payload + oligo.primer_rev_rc
    return Oligo(
        index=oligo.index,
        payload=new_payload,
        start_base=oligo.start_base,
        full_seq=new_full,
        primer_fwd=oligo.primer_fwd,
        primer_rev_rc=oligo.primer_rev_rc,
        tm_fwd=oligo.tm_fwd,
        tm_rev=oligo.tm_rev,
        is_padded=oligo.is_padded,
    )


def _rebuild_oligo(
    ref: Oligo,
    index: int,
    payload: str,
    pool: OligoPool,
    is_parity: bool = False,
) -> Oligo:
    """Build a new oligo with *index* and *payload*, using *ref* for primers."""
    from dna_codec.codec.oligos import _int_to_bases, _FLAG_ENC

    index_field = _int_to_bases(index % (4**pool.index_len), pool.index_len)
    flags_field = _FLAG_ENC.get(ref.start_base, "AA")
    full_seq    = (
        ref.primer_fwd
        + index_field
        + flags_field
        + payload
        + ref.primer_rev_rc
    )
    return Oligo(
        index=index,
        payload=payload,
        start_base=ref.start_base,
        full_seq=full_seq,
        primer_fwd=ref.primer_fwd,
        primer_rev_rc=ref.primer_rev_rc,
        tm_fwd=ref.tm_fwd,
        tm_rev=ref.tm_rev,
        is_padded=is_parity,
    )
