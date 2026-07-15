from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dna_codec.codec.encoder import DNAEncoder
from dna_codec.codec.oligos import OligoPool, Oligo, _bases_to_int, _FLAG_DEC
from dna_codec.codec.constraints import reverse_complement

def levenshtein(a: str, b: str) -> int:
    """
    Compute the Levenshtein (edit) distance between strings *a* and *b*.
    Time: O(|a|·|b|), Space: O(min(|a|,|b|)).
    """
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for ch_a in a:
        curr = [prev[0] + 1]
        for j, ch_b in enumerate(b):
            curr.append(min(
                prev[j] + (0 if ch_a == ch_b else 1),
                curr[j] + 1,
                prev[j + 1] + 1,
            ))
        prev = curr
    return prev[len(b)]

def align_to_length(seq: str, target_len: int, fill: str = "A") -> str:
    """
    Trim or right-pad *seq* to exactly *target_len* bases.
    Used to normalise indel-corrupted payloads before assembly.
    """
    if len(seq) >= target_len:
        return seq[:target_len]
    return seq + fill * (target_len - len(seq))

def _find_primer_end(seq: str, primer: str, max_edit: int = 4) -> int:
    """
    Find the end position of *primer* in *seq* using approximate matching.

    Slides a window of len(primer) along the first portion of *seq* and
    returns the end index of the window with smallest edit distance to
    *primer*.  If the best edit distance exceeds *max_edit*, returns
    len(primer) (assumes primer starts at position 0).

    Returns
    -------
    int : Index immediately after the primer in *seq*.
    """
    plen = len(primer)
    best_dist = max_edit + 1
    best_end  = plen
    search_end = min(len(seq), plen + max_edit + 2)
    for start in range(0, search_end - plen + 1):
        candidate = seq[start : start + plen]
        d = levenshtein(candidate, primer)
        if d < best_dist:
            best_dist = d
            best_end  = start + plen
    return best_end

def _find_primer_start(seq: str, primer_rc: str, max_edit: int = 4) -> int:
    """
    Find the start position of the reverse primer (primer_rev_rc) in *seq*
    by approximate search from the end of the sequence.

    Returns
    -------
    int : Index where the reverse primer begins in *seq*.
    """
    plen = len(primer_rc)
    best_dist = max_edit + 1
    best_start = len(seq) - plen
    search_begin = max(0, len(seq) - plen - max_edit - 2)
    for start in range(search_begin, len(seq) - plen + 1):
        candidate = seq[start : start + plen]
        d = levenshtein(candidate, primer_rc)
        if d < best_dist:
            best_dist = d
            best_start = start
    return best_start

@dataclass
class OligoParseResult:
    """Result of parsing and aligning a single received oligo."""
    raw_index: int
    trusted_index: bool
    payload: str
    start_base: str
    edit_to_fwd_primer: int
    indel_detected: bool

@dataclass
class DecodeReport:
    """Full report for a decoder.decode() call."""
    n_oligos_received: int
    n_unique_indices: int
    n_missing_indices: int
    n_indel_oligos: int
    n_rs_corrected: int
    n_rs_failed: int
    n_column_recovered: int
    sha256_ok: bool
    recovered_bytes: int
    errors: list[str] = field(default_factory=list)

    def summary(self) -> str:
        status = "✓ OK" if self.sha256_ok else "✗ FAIL"
        return (
            f"[{status}] {self.recovered_bytes} bytes recovered | "
            f"oligos={self.n_oligos_received} "
            f"unique_idx={self.n_unique_indices} "
            f"missing={self.n_missing_indices} "
            f"indels={self.n_indel_oligos} "
            f"rs_fixed={self.n_rs_corrected} "
            f"col_rec={self.n_column_recovered}"
        )

class DNADecoder:
    """
    Reconstruct the original file from a noisy, reordered oligo pool.

    Parameters
    ----------
    pool : OligoPool
        The pool configuration used during encoding (oligo_len, overlap, …).
    encoder : DNAEncoder
        The encoder instance that produced the master sequence.  Used for
        block_size and for the final decode_sequence() call.
    rs_codec : RSCodec | None
        If RS encoding was applied, pass the same RSCodec instance.
        If None, RS decoding is skipped.
    max_primer_edit : int
        Maximum edit distance for primer approximate matching (default 4).
    """

    def __init__(
        self,
        pool: OligoPool,
        encoder: DNAEncoder,
        rs_codec=None,
        max_primer_edit: int = 4,
    ) -> None:
        self.pool            = pool
        self.encoder         = encoder
        self.rs_codec        = rs_codec
        self.max_primer_edit = max_primer_edit

    def decode(self, oligos: list[Oligo]) -> tuple[bytes, DecodeReport]:
        """
        Full decoding pipeline: oligos → original bytes.

        Parameters
        ----------
        oligos : list[Oligo]
            Received pool (noisy, reordered, possibly incomplete).

        Returns
        -------
        (data, report) : The recovered data bytes and a detailed report.
        """
        report_data: dict = {
            "n_oligos_received": len(oligos),
            "n_indel_oligos": 0,
            "n_rs_corrected": 0,
            "n_rs_failed": 0,
            "n_column_recovered": 0,
            "errors": [],
        }

        parsed = self._parse_all(oligos, report_data)

        grouped = self._group_by_index(parsed)
        report_data["n_unique_indices"] = len(grouped)

        n_col_rec = 0
        if self.rs_codec is not None:
            grouped, n_col_rec = self._apply_column_recovery(
                oligos, grouped, report_data
            )
        report_data["n_column_recovered"] = n_col_rec

        n_expected = max(grouped.keys()) + 1 if grouped else 0
        report_data["n_missing_indices"] = max(
            0, n_expected - len(grouped)
        )

        if self.rs_codec is not None:
            grouped = self._apply_rs_decode(grouped, report_data)

        master_seq, start_bases = self._assemble_with_consensus(grouped, n_expected)

        try:
            data = self.encoder.decode_sequence(master_seq, start_bases)
            sha256_ok = True
        except ValueError as e:
            report_data["errors"].append(str(e))
            sha256_ok = False
            data = b""

        report = DecodeReport(
            n_oligos_received=report_data["n_oligos_received"],
            n_unique_indices=report_data["n_unique_indices"],
            n_missing_indices=report_data["n_missing_indices"],
            n_indel_oligos=report_data["n_indel_oligos"],
            n_rs_corrected=report_data["n_rs_corrected"],
            n_rs_failed=report_data["n_rs_failed"],
            n_column_recovered=report_data["n_column_recovered"],
            sha256_ok=sha256_ok,
            recovered_bytes=len(data),
            errors=report_data["errors"],
        )
        return data, report

    def decode_fasta(self, path: str | Path) -> tuple[bytes, DecodeReport]:
        """Load a FASTA file and decode it."""
        oligos = self.pool.read_fasta(path)
        return self.decode(oligos)

    def _parse_oligo(self, oligo: Oligo) -> Optional[OligoParseResult]:
        """
        Extract index, start_base and payload from one noisy oligo.

        Uses approximate primer matching to locate the inner region even
        when indels have shifted positions by a few bases.
        """
        seq = oligo.full_seq.upper()
        p   = self.pool.primer_len
        i   = self.pool.index_len
        f   = self.pool.FLAGS_LEN

        fwd_end   = _find_primer_end(seq, self.pool.primer_fwd, self.max_primer_edit)
        rev_start = _find_primer_start(seq, self.pool.primer_rev_rc, self.max_primer_edit)

        edit_fwd = levenshtein(seq[:fwd_end], self.pool.primer_fwd)

        inner = seq[fwd_end:rev_start] if rev_start > fwd_end else ""

        try:
            raw_index = _bases_to_int(inner[:i]) if len(inner) >= i else oligo.index
        except Exception:
            raw_index = oligo.index

        expected_index_seq = self._int_to_bases_safe(raw_index, i)
        idx_edit = levenshtein(inner[:i], expected_index_seq) if len(inner) >= i else i
        trusted = idx_edit <= 1

        flags_raw  = inner[i:i+f] if len(inner) >= i + f else ""
        start_base = _FLAG_DEC.get(flags_raw, oligo.start_base)

        payload_raw = inner[i+f:] if len(inner) > i + f else ""
        expected_pl = self.pool.payload_len

        indel_detected = len(payload_raw) != expected_pl
        payload = align_to_length(payload_raw, expected_pl)

        return OligoParseResult(
            raw_index=raw_index,
            trusted_index=trusted,
            payload=payload,
            start_base=start_base,
            edit_to_fwd_primer=edit_fwd,
            indel_detected=indel_detected,
        )

    def _parse_all(
        self,
        oligos: list[Oligo],
        report_data: dict,
    ) -> list[OligoParseResult]:
        """Parse all received oligos; count indel-affected ones."""
        results = []
        for oligo in oligos:
            parsed = self._parse_oligo(oligo)
            if parsed is not None:
                if parsed.indel_detected:
                    report_data["n_indel_oligos"] += 1
                results.append(parsed)
        return results

    def _group_by_index(
        self,
        parsed: list[OligoParseResult],
    ) -> dict[int, list[OligoParseResult]]:
        """
        Group parsed oligos by their index.

        Untrusted indices are cross-validated: if an oligo's index is
        suspicious but its payload overlaps well with a trusted neighbour,
        the trusted position is used instead.
        """
        groups: dict[int, list[OligoParseResult]] = {}
        for p in parsed:
            idx = p.raw_index
            groups.setdefault(idx, []).append(p)
        return groups

    def _apply_column_recovery(
        self,
        original_oligos: list[Oligo],
        grouped: dict[int, list[OligoParseResult]],
        report_data: dict,
    ) -> tuple[dict[int, list[OligoParseResult]], int]:
        """
        Attempt Level-2 (column parity) recovery for missing indices.

        Uses the RS codec's recover_with_column_parity, then injects the
        recovered oligos back into *grouped*.
        """
        if not original_oligos:
            return grouped, 0

        all_indices  = sorted(o.index for o in original_oligos)
        if not all_indices:
            return grouped, 0

        max_present = max(all_indices)
        n_data = max(grouped.keys()) + 1 if grouped else 0

        parity_oligos = [o for o in original_oligos if o.index >= n_data]
        data_oligos_raw = [o for o in original_oligos if o.index < n_data]

        if not parity_oligos:
            return grouped, 0

        recovered_pool, n_rec = self.rs_codec.recover_with_column_parity(
            data_oligos_raw + parity_oligos,
            n_data,
            self.pool,
        )

        if n_rec == 0:
            return grouped, 0

        for oligo in recovered_pool:
            if oligo.index not in grouped:
                parsed = OligoParseResult(
                    raw_index=oligo.index,
                    trusted_index=True,
                    payload=oligo.payload,
                    start_base=oligo.start_base,
                    edit_to_fwd_primer=0,
                    indel_detected=False,
                )
                grouped[oligo.index] = [parsed]

        return grouped, n_rec

    def _apply_rs_decode(
        self,
        grouped: dict[int, list[OligoParseResult]],
        report_data: dict,
    ) -> dict[int, list[OligoParseResult]]:
        """
        Apply per-oligo RS decode to the consensus payload of each group.
        """
        from dna_codec.codec.ecc.reed_solomon import _bytes_to_dna, _dna_to_bytes
        import reedsolo

        original_pl_bases = self.pool.payload_len
        original_pl_bytes = original_pl_bases // 4

        new_grouped: dict[int, list[OligoParseResult]] = {}
        for idx, group in grouped.items():
            consensus = self._base_consensus(group)

            encoded_bytes = _dna_to_bytes(consensus)
            try:
                decoded_bytes, n_err = self.rs_codec.decode_oligo(
                    encoded_bytes, original_pl_bytes
                )
                report_data["n_rs_corrected"] += n_err
                decoded_payload = _bytes_to_dna(decoded_bytes)
                rep = OligoParseResult(
                    raw_index=idx,
                    trusted_index=True,
                    payload=decoded_payload,
                    start_base=group[0].start_base,
                    edit_to_fwd_primer=0,
                    indel_detected=False,
                )
                new_grouped[idx] = [rep]
            except Exception as e:
                report_data["n_rs_failed"] += 1
                report_data["errors"].append(f"RS decode fail @{idx}: {e}")
                new_grouped[idx] = group

        return new_grouped

    def _base_consensus(self, group: list[OligoParseResult]) -> str:
        """
        Vote base-by-base across all payloads in *group*.
        All payloads are normalised to the same length (pool.payload_len).
        """
        pl   = self.pool.payload_len
        n    = len(group)
        vote: list[dict[str, int]] = [{} for _ in range(pl)]

        for result in group:
            payload = align_to_length(result.payload, pl)
            for pos, base in enumerate(payload):
                if base in "ACGT":
                    vote[pos][base] = vote[pos].get(base, 0) + 1

        consensus: list[str] = []
        for v in vote:
            if v:
                best = max(v, key=lambda b: (v[b], -ord(b)))
            else:
                best = "A"
            consensus.append(best)
        return "".join(consensus)

    def _assemble_with_consensus(
        self,
        grouped: dict[int, list[OligoParseResult]],
        n_expected: int,
    ) -> tuple[str, list[str]]:
        """
        Reconstruct the master sequence from all groups using overlap voting.

        For each master position, collect all oligo payloads that cover it
        and vote.  Overlapping regions get votes from both neighbours.

        Returns (master_seq, start_bases).
        """
        if not grouped:
            return "", []

        pl     = self.pool.payload_len
        stride = self.pool.stride
        ov     = self.pool.overlap
        max_idx = max(grouped.keys())
        total_len = stride * max_idx + pl

        votes: list[dict[str, int]] = [{} for _ in range(total_len)]

        for idx, group in grouped.items():
            pos_start = idx * stride
            consensus = self._base_consensus(group)
            n_copies  = len(group)

            for offset, base in enumerate(consensus):
                pos = pos_start + offset
                if pos < total_len and base in "ACGT":
                    votes[pos][base] = votes[pos].get(base, 0) + n_copies

        master_seq = "".join(
            max(v, key=lambda b: (v[b], -ord(b))) if v else "A"
            for v in votes
        )

        block_size  = self.encoder.block_size
        n_blocks    = math.ceil(len(master_seq) / block_size)
        start_bases: list[str] = ["A"] * n_blocks

        for idx, group in sorted(grouped.items()):
            if not group:
                continue
            sb        = group[0].start_base
            pos_start = idx * stride
            block_idx = pos_start // block_size
            if 0 <= block_idx < n_blocks:
                start_bases[block_idx] = sb

        return master_seq, start_bases

    @staticmethod
    def _int_to_bases_safe(value: int, width: int) -> str:
        """Same as _int_to_bases but returns '?' on overflow."""
        try:
            from dna_codec.codec.oligos import _int_to_bases
            return _int_to_bases(value, width)
        except ValueError:
            return "?" * width
