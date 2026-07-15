from __future__ import annotations

import os
import random

import pytest

from dna_codec.codec.encoder import DNAEncoder
from dna_codec.codec.oligos import OligoPool, Oligo
from dna_codec.codec.ecc.reed_solomon import (
    RSCodec,
    PoolDecodeReport,
    _bytes_to_dna,
    _dna_to_bytes,
)
from dna_codec.channel.simulator import ChannelSimulator, ChannelStats

@pytest.fixture
def encoder() -> DNAEncoder:
    return DNAEncoder(block_size=200)

@pytest.fixture
def pool() -> OligoPool:
    return OligoPool(oligo_len=150, overlap=20, primer_len=20, index_len=8)

@pytest.fixture
def rs() -> RSCodec:
    return RSCodec(redundancy=0.30, col_redundancy=0.20)

@pytest.fixture
def sim() -> ChannelSimulator:
    return ChannelSimulator(
        sub_rate=0.01, ins_rate=0.005, del_rate=0.005,
        dropout_rate=0.05, seed=42,
    )

def make_oligos(n_bytes: int = 300):
    enc  = DNAEncoder(block_size=200)
    pool = OligoPool(oligo_len=150, overlap=20, primer_len=20, index_len=8)
    data = os.urandom(n_bytes)
    master = enc.encode_bytes(data)
    oligos = pool.fragment(master, enc.start_bases)
    return oligos, enc, pool, data

class TestDNABytesHelpers:
    def test_bytes_to_dna_zero(self):
        assert _bytes_to_dna(b"\x00") == "AAAA"

    def test_bytes_to_dna_ff(self):
        assert _bytes_to_dna(b"\xFF") == "TTTT"

    def test_round_trip_single_byte(self):
        for b in range(256):
            data = bytes([b])
            assert _dna_to_bytes(_bytes_to_dna(data)) == data

    def test_round_trip_random(self):
        data = os.urandom(64)
        assert _dna_to_bytes(_bytes_to_dna(data)) == data

    def test_dna_to_bytes_truncates_partial(self):
        result = _dna_to_bytes("ACGTA")
        assert len(result) == 1

class TestRSPerOligo:
    def test_encode_increases_length(self, rs: RSCodec):
        payload = os.urandom(25)
        encoded = rs.encode_oligo(payload)
        assert len(encoded) > len(payload)
        assert len(encoded) == len(payload) + rs.nsym_for(len(payload))

    def test_clean_round_trip(self, rs: RSCodec):
        for _ in range(20):
            payload = os.urandom(25)
            encoded = rs.encode_oligo(payload)
            recovered, n_err = rs.decode_oligo(encoded, len(payload))
            assert recovered == payload
            assert n_err == 0

    def test_corrects_single_error(self, rs: RSCodec):
        payload = os.urandom(25)
        encoded = bytearray(rs.encode_oligo(payload))
        encoded[5] ^= 0x55
        recovered, n_err = rs.decode_oligo(bytes(encoded), len(payload))
        assert recovered == payload
        assert n_err >= 1

    def test_corrects_up_to_capacity(self, rs: RSCodec):
        payload = os.urandom(25)
        encoded = bytearray(rs.encode_oligo(payload))
        nsym    = rs.nsym_for(len(payload))
        capacity = nsym // 2

        positions = random.sample(range(len(encoded)), capacity)
        for pos in positions:
            encoded[pos] ^= 0xFF
        recovered, n_err = rs.decode_oligo(bytes(encoded), len(payload))
        assert recovered == payload

    def test_fails_beyond_capacity(self, rs: RSCodec):
        """Too many errors → ReedSolomonError or garbled output."""
        import reedsolo
        payload = os.urandom(25)
        encoded = bytearray(rs.encode_oligo(payload))
        nsym    = rs.nsym_for(len(payload))
        positions = list(range(nsym + 1))
        for pos in positions:
            if pos < len(encoded):
                encoded[pos] ^= 0xFF
        with pytest.raises(reedsolo.ReedSolomonError):
            rs.decode_oligo(bytes(encoded), len(payload))

    def test_erasure_decoding(self, rs: RSCodec):
        """Erasures (known positions) correct twice as many errors."""
        payload = os.urandom(25)
        encoded = bytearray(rs.encode_oligo(payload))
        nsym    = rs.nsym_for(len(payload))
        erasure_positions = list(range(len(payload), len(encoded)))
        for pos in erasure_positions:
            encoded[pos] = 0
        recovered, _ = rs.decode_oligo(
            bytes(encoded), len(payload), erasures=erasure_positions
        )
        assert recovered == payload

class TestRSPool:
    def test_encode_pool_extends_payload(self, rs: RSCodec):
        oligos, enc, pool, data = make_oligos(200)
        original_payload_len = len(oligos[0].payload)
        encoded = rs.encode_pool(oligos)
        nsym = rs.nsym_for(original_payload_len // 4)
        expected = original_payload_len + nsym * 4
        for o in encoded:
            assert len(o.payload) == expected, f"Oligo {o.index}: {len(o.payload)} != {expected}"

    def test_clean_pool_round_trip(self, rs: RSCodec):
        oligos, enc, pool, data = make_oligos(200)
        original_pl = len(oligos[0].payload)

        encoded = rs.encode_pool(oligos)
        decoded, report = rs.decode_pool(encoded, original_payload_bases=original_pl)

        assert report.decoded_ok == len(oligos)
        assert report.failed == 0
        for orig, dec in zip(oligos, decoded):
            assert dec.payload == orig.payload

    def test_decode_pool_with_errors(self, rs: RSCodec):
        """Simulate base errors in DNA payloads and verify RS recovery."""
        oligos, enc, pool, data = make_oligos(200)
        original_pl = len(oligos[0].payload)
        encoded = rs.encode_pool(oligos)

        import random as rnd
        rnd.seed(7)
        corrupted = list(encoded)
        for i in range(min(3, len(corrupted))):
            pos = rnd.randint(0, len(corrupted[i].payload) - 1)
            bad_base = rnd.choice(
                [b for b in "ACGT" if b != corrupted[i].payload[pos]]
            )
            pl = list(corrupted[i].payload)
            pl[pos] = bad_base
            from dna_codec.codec.ecc.reed_solomon import _replace_payload
            corrupted[i] = _replace_payload(corrupted[i], "".join(pl))

        decoded, report = rs.decode_pool(corrupted, original_payload_bases=original_pl)
        assert report.decoded_ok == len(oligos)

    def test_report_has_correct_structure(self, rs: RSCodec):
        oligos, enc, pool, data = make_oligos(100)
        original_pl = len(oligos[0].payload)
        encoded = rs.encode_pool(oligos)
        _, report = rs.decode_pool(encoded, original_pl)

        assert hasattr(report, "total_oligos")
        assert hasattr(report, "success_rate")
        assert 0.0 <= report.success_rate <= 1.0
        assert len(report.oligo_results) == len(oligos)

class TestRSColumnParity:
    def test_column_parity_adds_oligos(self, rs: RSCodec, pool: OligoPool):
        oligos, enc, pl, data = make_oligos(300)
        extended = rs.add_column_parity(oligos, pool)
        n_parity_expected = max(2, round(len(oligos) * rs.col_redundancy))
        assert len(extended) == len(oligos) + n_parity_expected

    def test_parity_oligos_correct_length(self, rs: RSCodec, pool: OligoPool):
        oligos, enc, pl, data = make_oligos(300)
        extended = rs.add_column_parity(oligos, pool)
        expected_len = pool.oligo_len
        for o in extended:
            assert abs(len(o.full_seq) - expected_len) <= 8, (
                f"Oligo {o.index} full_seq len={len(o.full_seq)}"
            )

    def test_column_parity_recovery_single_dropout(self, rs: RSCodec, pool: OligoPool):
        """Drop one data oligo → column parity recovers it perfectly."""
        oligos, enc, pl, data = make_oligos(300)
        n_data = len(oligos)
        extended = rs.add_column_parity(oligos, pool)

        without_oligo2 = [o for o in extended if o.index != 2]

        recovered, n_rec = rs.recover_with_column_parity(
            without_oligo2, n_data, pool
        )
        assert n_rec >= 1
        assert len(recovered) == n_data

        by_idx = {o.index: o for o in recovered}
        assert 2 in by_idx
        assert by_idx[2].payload == oligos[2].payload

    def test_column_parity_recovery_multiple_dropouts(self, rs: RSCodec, pool: OligoPool):
        """Drop several oligos within parity capacity."""
        oligos, enc, pl, data = make_oligos(500)
        n_data = len(oligos)
        extended = rs.add_column_parity(oligos, pool)
        nsym = max(2, round(n_data * rs.col_redundancy))

        drop_count = max(1, nsym - 1)
        drop_indices = sorted(random.sample(range(n_data), drop_count))
        surviving = [o for o in extended if o.index not in drop_indices]

        recovered, n_rec = rs.recover_with_column_parity(surviving, n_data, pool)
        assert n_rec == drop_count
        assert len(recovered) == n_data

        by_idx = {o.index: o for o in recovered}
        for di in drop_indices:
            assert di in by_idx
            assert by_idx[di].payload == oligos[di].payload

class TestChannelSimulator:
    def test_clean_channel_no_errors(self):
        """With all rates = 0, sequences should be unchanged."""
        sim = ChannelSimulator(sub_rate=0, ins_rate=0, del_rate=0,
                               dropout_rate=0, seed=1)
        oligos, *_ = make_oligos(200)
        noisy, stats = sim.simulate(oligos, reorder=False)
        assert len(noisy) == len(oligos)
        assert stats.total_errors == 0
        assert stats.n_dropped == 0

    def test_substitution_rate_approximate(self):
        """Observed substitution rate should be close to configured rate."""
        target = 0.05
        sim = ChannelSimulator(sub_rate=target, ins_rate=0, del_rate=0,
                               dropout_rate=0, seed=99)
        seq = "ACGT" * 2500
        _, stats = sim.corrupt_sequence(seq)
        observed = stats.n_substitutions / len(seq)
        assert abs(observed - target) < 0.02, (
            f"Sub rate {observed:.3f} too far from {target}"
        )

    def test_insertions_increase_length(self):
        sim = ChannelSimulator(sub_rate=0, ins_rate=0.1, del_rate=0,
                               dropout_rate=0, seed=5)
        seq = "ACGT" * 250
        noisy, stats = sim.corrupt_sequence(seq)
        assert stats.n_insertions > 0
        assert len(noisy) > len(seq)

    def test_deletions_decrease_length(self):
        sim = ChannelSimulator(sub_rate=0, ins_rate=0, del_rate=0.1,
                               dropout_rate=0, seed=6)
        seq = "ACGT" * 250
        noisy, stats = sim.corrupt_sequence(seq)
        assert stats.n_deletions > 0
        assert len(noisy) < len(seq)

    def test_dropout_reduces_pool(self):
        sim = ChannelSimulator(sub_rate=0, ins_rate=0, del_rate=0,
                               dropout_rate=0.3, seed=7)
        oligos, *_ = make_oligos(300)
        noisy, stats = sim.simulate(oligos, reorder=False)
        assert stats.n_dropped > 0
        assert len(noisy) < len(oligos)
        assert stats.dropout_fraction == pytest.approx(
            stats.n_dropped / len(oligos)
        )

    def test_reordering_changes_order(self):
        sim = ChannelSimulator(sub_rate=0, ins_rate=0, del_rate=0,
                               dropout_rate=0, seed=8)
        oligos, *_ = make_oligos(500)
        noisy, _   = sim.simulate(oligos, reorder=True)
        original_indices = [o.index for o in oligos]
        noisy_indices    = [o.index for o in noisy]
        assert set(noisy_indices) == set(original_indices)
        assert noisy_indices != original_indices

    def test_no_reorder(self):
        sim = ChannelSimulator(sub_rate=0, ins_rate=0, del_rate=0,
                               dropout_rate=0, seed=9)
        oligos, *_ = make_oligos(200)
        noisy, _   = sim.simulate(oligos, reorder=False)
        assert [o.index for o in noisy] == [o.index for o in oligos]

    def test_stats_snr_positive(self):
        sim = ChannelSimulator(sub_rate=0.01, ins_rate=0, del_rate=0,
                               dropout_rate=0, seed=10)
        oligos, *_ = make_oligos(300)
        _, stats = sim.simulate(oligos, reorder=False)
        assert stats.snr_db > 0

    def test_stats_summary_string(self):
        sim = ChannelSimulator(sub_rate=0.01, ins_rate=0.005, del_rate=0.005,
                               dropout_rate=0.05, seed=11)
        oligos, *_ = make_oligos(300)
        _, stats = sim.simulate(oligos)
        summary = stats.summary()
        assert "Channel:" in summary
        assert "SNR=" in summary

    def test_dropout_only(self):
        sim = ChannelSimulator(sub_rate=0.05, ins_rate=0.05, del_rate=0.05,
                               dropout_rate=0.2, seed=12)
        oligos, *_ = make_oligos(300)
        noisy, stats = sim.simulate_dropout_only(oligos)
        assert stats.total_substitutions == 0
        assert stats.total_insertions == 0
        assert stats.total_deletions == 0
        assert stats.n_dropped > 0

    def test_reproducible_with_seed(self):
        oligos, *_ = make_oligos(200)
        sim1 = ChannelSimulator(sub_rate=0.01, dropout_rate=0.05, seed=42)
        sim2 = ChannelSimulator(sub_rate=0.01, dropout_rate=0.05, seed=42)
        n1, s1 = sim1.simulate(oligos, reorder=True)
        n2, s2 = sim2.simulate(oligos, reorder=True)
        assert [o.index for o in n1] == [o.index for o in n2]
        assert s1.n_dropped == s2.n_dropped

class TestIntegrationRS:
    def test_rs_protects_against_substitutions(self):
        """
        Encode with RS → pass through noisy channel → RS decode → verify
        payload is recovered despite substitution errors.
        """
        enc  = DNAEncoder(block_size=200)
        pl   = OligoPool(oligo_len=150, overlap=20, primer_len=20, index_len=8)
        rs   = RSCodec(redundancy=0.30)
        data = os.urandom(200)

        master = enc.encode_bytes(data)
        oligos = pl.fragment(master, enc.start_bases)
        orig_pl_len = len(oligos[0].payload)

        encoded_oligos = rs.encode_pool(oligos)

        sim   = ChannelSimulator(sub_rate=0.005, ins_rate=0, del_rate=0,
                                 dropout_rate=0, seed=77)
        noisy, chan_stats = sim.simulate(encoded_oligos, reorder=True)

        assert chan_stats.total_substitutions > 0

        decoded_oligos, report = rs.decode_pool(noisy, orig_pl_len)
        assert report.decoded_ok == len(oligos)
        assert report.failed == 0

        decoded_oligos.sort(key=lambda o: o.index)
        rebuilt_master, _ = pl.assemble(decoded_oligos)
        recovered = enc.decode_sequence(rebuilt_master[:len(master)], enc.start_bases)
        assert recovered == data

    def test_rs_2d_recovers_dropped_oligos(self):
        """
        Column parity allows recovery of dropped oligos — simulating
        real DNA sequencing where some sequences are never read.
        """
        enc  = DNAEncoder(block_size=200)
        pl   = OligoPool(oligo_len=150, overlap=20, primer_len=20, index_len=8)
        rs   = RSCodec(redundancy=0.20, col_redundancy=0.25)
        data = os.urandom(400)

        master = enc.encode_bytes(data)
        oligos = pl.fragment(master, enc.start_bases)
        n_data = len(oligos)

        extended = rs.add_column_parity(oligos, pl)

        drop_idx = [1, 3]
        surviving = [o for o in extended if o.index not in drop_idx]

        recovered_pool, n_rec = rs.recover_with_column_parity(
            surviving, n_data, pl
        )
        assert n_rec == 2

        recovered_pool.sort(key=lambda o: o.index)
        rebuilt_master, _ = pl.assemble(recovered_pool)
        recovered = enc.decode_sequence(rebuilt_master[:len(master)], enc.start_bases)
        assert recovered == data

    @pytest.mark.slow
    def test_full_pipeline_realistic_channel(self):
        """
        Realistic channel with dropout (10%) and reordering.
        RS Level-1 is already tested in test_rs_protects_against_substitutions.
        RS Level-2 (column parity) handles dropout; index-field substitution
        corruption is handled by Stage 4's alignment decoder.

        Note: with sub_rate > 0, substitutions in the 8-base index field
        (not RS-protected) would misplace oligos in assembly — that is
        corrected in Stage 4 via sequence alignment and consensus voting.
        """
        enc  = DNAEncoder(block_size=200)
        pl   = OligoPool(oligo_len=150, overlap=20, primer_len=20, index_len=8)
        rs   = RSCodec(redundancy=0.30, col_redundancy=0.30)
        data = os.urandom(512)

        master = enc.encode_bytes(data)
        oligos = pl.fragment(master, enc.start_bases)
        n_data = len(oligos)
        orig_pl_len = len(oligos[0].payload)

        encoded = rs.encode_pool(oligos)

        extended = rs.add_column_parity(encoded, pl)

        sim   = ChannelSimulator(sub_rate=0.0, ins_rate=0.0, del_rate=0.0,
                                 dropout_rate=0.10, seed=2024)
        noisy, chan_stats = sim.simulate(extended, reorder=True)
        assert chan_stats.n_dropped > 0
        assert chan_stats.n_oligos_out < chan_stats.n_oligos_in

        noisy.sort(key=lambda o: o.index)
        data_noisy   = [o for o in noisy if o.index < n_data]
        parity_noisy = [o for o in noisy if o.index >= n_data]

        recovered_data, n_rec = rs.recover_with_column_parity(
            data_noisy + parity_noisy, n_data, pl
        )
        assert len(recovered_data) == n_data

        decoded, report = rs.decode_pool(recovered_data, orig_pl_len)
        assert report.decoded_ok == n_data

        decoded.sort(key=lambda o: o.index)
        rebuilt, _ = pl.assemble(decoded)
        final = enc.decode_sequence(rebuilt[:len(master)], enc.start_bases)
        assert final == data

class TestColumnParityRegressions:
    """
    Regression tests for two bugs in the Level-2 (column) RS parity path,
    both surfaced when running a realistic image (>255 data oligos)
    through the full pipeline:

      1. add_column_parity built a single RS codeword spanning all N data
         oligos. GF(2^8) limits a codeword to 255 symbols (data + parity),
         so once N + nsym > 255 reedsolo produced a truncated/corrupt
         result instead of raising, surfacing later as an IndexError.

      2. recover_with_column_parity computed nsym from the number of
         *surviving* parity oligos rather than the number the codeword
         was originally built with, and passed erase_pos using indices
         relative only to the data portion instead of the full codeword
         (data + parity). Losing any parity oligo alongside data oligos
         therefore silently corrupted recovery even when total erasures
         were within RS's correction capacity.
    """

    def test_add_column_parity_handles_large_pools(self):
        """N + nsym > 255 must not raise and must produce a pool whose
        column codewords are still individually decodable."""
        enc = DNAEncoder(block_size=200)
        pl = OligoPool(oligo_len=150, overlap=20, primer_len=20, index_len=8)
        rs = RSCodec(redundancy=0.30, col_redundancy=0.20)

        data = os.urandom(15000)
        master = enc.encode_bytes(data)
        oligos = pl.fragment(master, enc.start_bases)
        n_data = len(oligos)
        assert n_data > 255, "test setup should exceed the GF(2^8) limit"

        encoded = rs.encode_pool(oligos)
        extended = rs.add_column_parity(encoded, pl)
        assert len(extended) > len(encoded)

        decoded, report = rs.decode_pool(
            [o for o in extended if o.index < n_data],
            original_payload_bases=len(oligos[0].payload),
        )
        assert report.decoded_ok == n_data
        decoded.sort(key=lambda o: o.index)
        rebuilt, _ = pl.assemble(decoded)
        final = enc.decode_sequence(rebuilt[:len(master)], enc.start_bases)
        assert final == data

    def test_recover_with_column_parity_survives_lost_parity_oligo(self):
        """Losing a data oligo AND a parity oligo simultaneously (total
        erasures still <= nsym) must still recover exactly, matching the
        original bug-report scenario (30 data oligos, col_redundancy=0.30
        -> nsym=9, 6 data oligos + 1 parity oligo lost)."""
        enc = DNAEncoder(block_size=200)
        pl = OligoPool(oligo_len=150, overlap=20, primer_len=20, index_len=8)
        rs = RSCodec(redundancy=0.30, col_redundancy=0.30)

        data = os.urandom(400)
        master = enc.encode_bytes(data)
        oligos = pl.fragment(master, enc.start_bases)
        n_data = len(oligos)

        extended = rs.add_column_parity(oligos, pl)
        n_parity = len(extended) - n_data
        assert n_parity >= 2, "need at least 2 parity oligos for this test"

        dropped_data_idx = 1
        dropped_parity_idx = n_data
        surviving = [
            o for o in extended
            if o.index != dropped_data_idx and o.index != dropped_parity_idx
        ]

        recovered_pool, n_rec = rs.recover_with_column_parity(
            surviving, n_data, pl
        )
        assert n_rec == 1
        assert len(recovered_pool) == n_data

        recovered_pool.sort(key=lambda o: o.index)
        rebuilt, _ = pl.assemble(recovered_pool)
        final = enc.decode_sequence(rebuilt[:len(master)], enc.start_bases)
        assert final == data

    def test_recover_within_capacity_across_seeds(self):
        """Property-style check: whenever total erasures (missing data +
        missing parity oligos) stay within nsym, recovery must succeed,
        across a range of random-dropout channel seeds."""
        enc = DNAEncoder(block_size=200)
        pl = OligoPool(oligo_len=150, overlap=20, primer_len=20, index_len=8)
        rs = RSCodec(redundancy=0.30, col_redundancy=0.30)
        data = os.urandom(300)

        master = enc.encode_bytes(data)
        oligos = pl.fragment(master, enc.start_bases)
        n_data = len(oligos)

        encoded = rs.encode_pool(oligos)
        extended = rs.add_column_parity(encoded, pl)
        n_parity = len(extended) - len(encoded)
        orig_pl_len = len(oligos[0].payload)

        successes_within_capacity = 0
        for seed in range(2, 20):
            sim = ChannelSimulator(
                sub_rate=0.0, ins_rate=0.0, del_rate=0.0,
                dropout_rate=0.15, seed=seed,
            )
            noisy, _ = sim.simulate(extended, reorder=True)
            noisy.sort(key=lambda o: o.index)
            data_n = [o for o in noisy if o.index < n_data]
            parity_n = [o for o in noisy if o.index >= n_data]
            total_erasures = (n_data - len(data_n)) + (n_parity - len(parity_n))

            recovered_pool, _ = rs.recover_with_column_parity(
                data_n + parity_n, n_data, pl
            )
            decoded, report = rs.decode_pool(recovered_pool, orig_pl_len)
            decoded.sort(key=lambda o: o.index)
            rebuilt, _ = pl.assemble(decoded)

            if total_erasures <= n_parity:
                final = enc.decode_sequence(rebuilt[:len(master)], enc.start_bases)
                assert final == data, (
                    f"seed={seed}: recovery should have succeeded "
                    f"(erasures={total_erasures} <= nsym={n_parity})"
                )
                successes_within_capacity += 1

        assert successes_within_capacity > 0, "no seed exercised the recovery path"
