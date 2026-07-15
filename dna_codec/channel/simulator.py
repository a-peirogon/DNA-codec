from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

from dna_codec.codec.oligos import Oligo

BASES = ["A", "C", "G", "T"]

@dataclass
class SeqStats:
    """Error statistics for a single sequence."""
    original_len: int
    final_len: int
    n_substitutions: int = 0
    n_insertions: int = 0
    n_deletions: int = 0

    @property
    def total_errors(self) -> int:
        return self.n_substitutions + self.n_insertions + self.n_deletions

    @property
    def error_rate(self) -> float:
        denom = max(self.original_len, 1)
        return self.total_errors / denom

@dataclass
class ChannelStats:
    """Aggregated statistics for a full pool simulation run."""
    n_oligos_in: int
    n_oligos_out: int
    n_dropped: int
    total_bases_in: int
    total_bases_out: int
    total_substitutions: int = 0
    total_insertions: int = 0
    total_deletions: int = 0
    per_oligo: list[SeqStats] = field(default_factory=list)

    sub_rate: float = 0.0
    ins_rate: float = 0.0
    del_rate: float = 0.0
    dropout_rate: float = 0.0

    @property
    def total_errors(self) -> int:
        return self.total_substitutions + self.total_insertions + self.total_deletions

    @property
    def dropout_fraction(self) -> float:
        return self.n_dropped / max(self.n_oligos_in, 1)

    @property
    def base_error_rate(self) -> float:
        denom = max(self.total_bases_in, 1)
        return self.total_errors / denom

    @property
    def snr_db(self) -> float:
        """
        Estimated SNR in dB.
        Signal = correctly transmitted bases; Noise = errors.
        SNR = 10 · log10(correct / error).
        """
        correct = self.total_bases_out - self.total_errors
        errors  = max(self.total_errors, 1)
        import math
        return 10 * math.log10(max(correct, 1) / errors)

    def summary(self) -> str:
        return (
            f"Channel: {self.n_oligos_out}/{self.n_oligos_in} oligos "
            f"({self.n_dropped} dropped, {self.dropout_fraction:.1%} loss)\n"
            f"  Sub={self.total_substitutions} "
            f"Ins={self.total_insertions} "
            f"Del={self.total_deletions} "
            f"| error_rate={self.base_error_rate:.3%} "
            f"| SNR={self.snr_db:.1f} dB"
        )

class ChannelSimulator:
    """
    Simulates a noisy DNA synthesis + sequencing channel.

    Parameters
    ----------
    sub_rate : float
        Probability of base substitution per base (default 0.01 = 1%).
    ins_rate : float
        Probability of a random base insertion after each base (default 0.005).
    del_rate : float
        Probability of base deletion per base (default 0.005).
    dropout_rate : float
        Probability that an entire oligo is lost from the pool (default 0.05).
    seed : int | None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        sub_rate: float = 0.01,
        ins_rate: float = 0.005,
        del_rate: float = 0.005,
        dropout_rate: float = 0.05,
        seed: Optional[int] = None,
    ) -> None:
        self.sub_rate     = sub_rate
        self.ins_rate     = ins_rate
        self.del_rate     = del_rate
        self.dropout_rate = dropout_rate
        self.rng = random.Random(seed)

    def corrupt_sequence(self, seq: str) -> tuple[str, SeqStats]:
        """
        Apply substitutions, insertions and deletions to a DNA sequence.

        Returns
        -------
        (noisy_seq, stats)
        """
        original_len = len(seq)
        stats = SeqStats(original_len=original_len, final_len=0)
        out: list[str] = []
        rng = self.rng

        for base in seq:
            if self.del_rate > 0 and rng.random() < self.del_rate:
                stats.n_deletions += 1
                continue

            if self.sub_rate > 0 and rng.random() < self.sub_rate:
                alt = rng.choice([b for b in BASES if b != base])
                out.append(alt)
                stats.n_substitutions += 1
            else:
                out.append(base)

            if self.ins_rate > 0 and rng.random() < self.ins_rate:
                out.append(rng.choice(BASES))
                stats.n_insertions += 1

        stats.final_len = len(out)
        return "".join(out), stats

    def simulate(
        self,
        oligos: list[Oligo],
        reorder: bool = True,
    ) -> tuple[list[Oligo], ChannelStats]:
        """
        Simulate the channel on a complete oligo pool.

        Steps applied in order:
          1. Dropout: remove each oligo with probability `dropout_rate`.
          2. Sequence corruption: substitutions, insertions, deletions.
          3. Reordering: shuffle the surviving pool.

        Parameters
        ----------
        oligos : list[Oligo]
            Ordered pool (as produced by OligoPool.fragment or encode_pool).
        reorder : bool
            Whether to shuffle the output pool (default True).  Set False
            for deterministic testing.

        Returns
        -------
        (noisy_oligos, stats)
        """
        n_in   = len(oligos)
        rng    = self.rng
        noisy: list[Oligo] = []
        seq_stats_list: list[SeqStats] = []
        n_dropped = 0

        for oligo in oligos:
            if self.dropout_rate > 0 and rng.random() < self.dropout_rate:
                n_dropped += 1
                continue

            noisy_seq, ss = self.corrupt_sequence(oligo.full_seq)
            seq_stats_list.append(ss)

            noisy_oligo = _make_noisy_oligo(oligo, noisy_seq)
            noisy.append(noisy_oligo)

        if reorder and noisy:
            rng.shuffle(noisy)

        stats = ChannelStats(
            n_oligos_in=n_in,
            n_oligos_out=len(noisy),
            n_dropped=n_dropped,
            total_bases_in=sum(s.original_len for s in seq_stats_list),
            total_bases_out=sum(s.final_len for s in seq_stats_list),
            total_substitutions=sum(s.n_substitutions for s in seq_stats_list),
            total_insertions=sum(s.n_insertions for s in seq_stats_list),
            total_deletions=sum(s.n_deletions for s in seq_stats_list),
            per_oligo=seq_stats_list,
            sub_rate=self.sub_rate,
            ins_rate=self.ins_rate,
            del_rate=self.del_rate,
            dropout_rate=self.dropout_rate,
        )
        return noisy, stats

    def simulate_dropout_only(
        self,
        oligos: list[Oligo],
        reorder: bool = True,
    ) -> tuple[list[Oligo], ChannelStats]:
        """Apply only oligo dropout (no base-level errors). Useful for testing."""
        saved = self.sub_rate, self.ins_rate, self.del_rate
        self.sub_rate = self.ins_rate = self.del_rate = 0.0
        result = self.simulate(oligos, reorder=reorder)
        self.sub_rate, self.ins_rate, self.del_rate = saved
        return result

def _make_noisy_oligo(original: Oligo, noisy_seq: str) -> Oligo:
    """
    Build a new Oligo from a corrupted full_seq string.

    The primer length is used to locate the inner region.  If the noisy
    sequence is shorter than expected (due to deletions), the payload is
    trimmed; if longer (insertions), it is used as-is and the decoder's
    alignment step will handle it.

    The `index` is re-read from the noisy inner region so the decoder gets
    the (possibly corrupted) index — this is realistic: if the index bases
    are corrupted, the decoder must either correct them via RS or discard
    the oligo.
    """
    from dna_codec.codec.oligos import _bases_to_int, _FLAG_DEC, _FLAG_ENC

    p = len(original.primer_fwd)
    i = 8
    f = 2

    inner_start = p
    inner_end   = len(noisy_seq) - p
    inner = noisy_seq[inner_start:inner_end] if inner_end > inner_start else ""

    try:
        idx = _bases_to_int(inner[:i]) if len(inner) >= i else original.index
    except Exception:
        idx = original.index

    flags_raw = inner[i:i+f] if len(inner) >= i + f else ""
    start_base = _FLAG_DEC.get(flags_raw, original.start_base)

    payload = inner[i+f:] if len(inner) > i + f else ""

    return Oligo(
        index=idx,
        payload=payload,
        start_base=start_base,
        full_seq=noisy_seq,
        primer_fwd=original.primer_fwd,
        primer_rev_rc=original.primer_rev_rc,
        tm_fwd=original.tm_fwd,
        tm_rev=original.tm_rev,
        is_padded=original.is_padded,
    )
