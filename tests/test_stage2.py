from __future__ import annotations

import os
import random
import tempfile
from pathlib import Path

import pytest

from dna_codec.codec.constraints import reverse_complement
from dna_codec.codec.encoder import DNAEncoder
from dna_codec.codec.oligos import (
    OligoPool,
    Oligo,
    _int_to_bases,
    _bases_to_int,
    _FLAG_DEC,
    _FLAG_ENC,
    tm_nearest_neighbor,
    validate_primer,
    design_primers,
)

@pytest.fixture
def pool() -> OligoPool:
    """Standard pool: 150 nt oligos, 20 nt overlap, 20 nt primers, 8 nt index."""
    return OligoPool(oligo_len=150, overlap=20, primer_len=20, index_len=8)

@pytest.fixture
def encoder() -> DNAEncoder:
    return DNAEncoder(block_size=200)

@pytest.fixture
def small_pool() -> OligoPool:
    """Smaller oligos for fast tests."""
    return OligoPool(oligo_len=100, overlap=10, primer_len=15, index_len=6)

class TestIndexEncoding:
    def test_zero(self):
        assert _int_to_bases(0, 4) == "AAAA"

    def test_one(self):
        assert _int_to_bases(1, 4) == "AAAC"

    def test_max(self):
        assert _int_to_bases(255, 4) == "TTTT"

    def test_round_trip(self):
        for width in [4, 6, 8]:
            for v in [0, 1, 42, 4**width - 1]:
                encoded = _int_to_bases(v, width)
                assert len(encoded) == width
                assert _bases_to_int(encoded) == v

    def test_overflow_raises(self):
        with pytest.raises(ValueError, match="too large"):
            _int_to_bases(256, 4)

    def test_random_round_trip(self):
        for _ in range(50):
            v = random.randint(0, 4**8 - 1)
            assert _bases_to_int(_int_to_bases(v, 8)) == v

class TestFlagsEncoding:
    def test_all_bases(self):
        for base in ["A", "C", "G", "T"]:
            encoded = _FLAG_ENC[base]
            assert len(encoded) == 2
            assert _FLAG_DEC[encoded] == base

    def test_unique_codes(self):
        codes = list(_FLAG_ENC.values())
        assert len(codes) == len(set(codes))

class TestTmNearestNeighbor:
    def test_typical_range(self):
        seq = "ACGTACGTACGTACGTACGT"
        tm = tm_nearest_neighbor(seq)
        assert 45.0 <= tm <= 80.0, f"Tm={tm:.1f} out of expected range"

    def test_gc_rich_higher_tm(self):
        at_seq = "ATATATATAT"
        gc_seq = "GCGCGCGCGC"
        assert tm_nearest_neighbor(gc_seq) > tm_nearest_neighbor(at_seq)

    def test_longer_higher_tm(self):
        short = "ACGTACGTAC"
        long_ = "ACGTACGTACGTACGTACGT"
        assert tm_nearest_neighbor(long_) > tm_nearest_neighbor(short)

    def test_single_base_returns_zero(self):
        assert tm_nearest_neighbor("A") == 0.0

    def test_empty_returns_zero(self):
        assert tm_nearest_neighbor("") == 0.0

class TestValidatePrimer:
    def test_good_primer(self):
        seq = "ACGTAGCTGATCGTACGAGT"
        ok, reasons = validate_primer(seq, tm_min=50.0, tm_max=75.0)
        assert ok, f"Expected valid primer, got reasons: {reasons}"

    def test_homopolymer_fails(self):
        seq = "A" * 20
        ok, reasons = validate_primer(seq)
        assert not ok
        assert any("homopolymer" in r or "GC" in r or "Tm" in r for r in reasons)

    def test_tm_too_low(self):
        seq = "ACGT"
        ok, reasons = validate_primer(seq, tm_min=55.0)
        assert not ok

    def test_all_gc_fails_tm_or_gc(self):
        seq = "GCGCGCGCGCGCGCGCGCGC"
        ok, reasons = validate_primer(seq)
        assert not ok

class TestDesignPrimers:
    def test_returns_two_strings(self, encoder: DNAEncoder):
        seq = encoder.encode_bytes(os.urandom(500))
        fwd, rev = design_primers(seq, primer_len=20)
        assert isinstance(fwd, str) and len(fwd) == 20
        assert isinstance(rev, str) and len(rev) == 20

    def test_primers_are_valid(self, encoder: DNAEncoder):
        seq = encoder.encode_bytes(os.urandom(500))
        fwd, rev = design_primers(seq, primer_len=20, tm_min=50.0, tm_max=80.0)
        ok_f, reasons_f = validate_primer(fwd, tm_min=50.0, tm_max=80.0)
        ok_r, reasons_r = validate_primer(rev, tm_min=50.0, tm_max=80.0)
        assert ok_f, f"fwd invalid: {reasons_f}"
        assert ok_r, f"rev invalid: {reasons_r}"

    def test_short_seed_uses_defaults(self):
        fwd, rev = design_primers("ACGT", primer_len=20)
        assert len(fwd) == 20
        assert len(rev) == 20

class TestOligoPoolConfig:
    def test_payload_len_calculated(self, pool: OligoPool):
        assert pool.payload_len == 100

    def test_stride_calculated(self, pool: OligoPool):
        assert pool.stride == 80

    def test_info_keys(self, pool: OligoPool):
        info = pool.info()
        for key in ["oligo_len", "payload_len", "overlap", "stride", "max_oligos"]:
            assert key in info

    def test_overhead_too_large_raises(self):
        with pytest.raises(ValueError, match="Overhead"):
            OligoPool(oligo_len=50, primer_len=30, index_len=8)

    def test_overlap_too_large_raises(self):
        with pytest.raises(ValueError, match="stride"):
            OligoPool(oligo_len=150, overlap=110, primer_len=20, index_len=8)

class TestFragment:
    def test_oligo_count(self, pool: OligoPool, encoder: DNAEncoder):
        data = os.urandom(200)
        master = encoder.encode_bytes(data)
        oligos = pool.fragment(master, encoder.start_bases)
        assert len(oligos) > 0

    def test_all_oligos_correct_length(self, pool: OligoPool, encoder: DNAEncoder):
        data = os.urandom(300)
        master = encoder.encode_bytes(data)
        oligos = pool.fragment(master, encoder.start_bases)
        for o in oligos:
            assert len(o.full_seq) == pool.oligo_len, (
                f"Oligo {o.index} has length {len(o.full_seq)}"
            )

    def test_oligo_structure(self, pool: OligoPool, encoder: DNAEncoder):
        """Verify primer_fwd | index | flags | payload | primer_rev_rc layout."""
        data = os.urandom(100)
        master = encoder.encode_bytes(data)
        oligos = pool.fragment(master, encoder.start_bases)

        for o in oligos:
            seq = o.full_seq
            p = pool.primer_len
            i = pool.index_len
            f = pool.FLAGS_LEN
            l = pool.payload_len

            assert seq[:p]       == pool.primer_fwd,    f"Oligo {o.index}: bad fwd primer"
            assert seq[-p:]      == pool.primer_rev_rc, f"Oligo {o.index}: bad rev primer"
            assert seq[p:p+i]    == _int_to_bases(o.index, i), f"Oligo {o.index}: bad index"
            assert seq[p+i:p+i+f] in _FLAG_DEC,        f"Oligo {o.index}: bad flags"
            assert seq[p+i+f:-p] == o.payload,          f"Oligo {o.index}: bad payload"

    def test_index_field_is_sequential(self, pool: OligoPool, encoder: DNAEncoder):
        data = os.urandom(500)
        master = encoder.encode_bytes(data)
        oligos = pool.fragment(master, encoder.start_bases)
        for i, o in enumerate(oligos):
            assert o.index == i

    def test_overlap_region_shared(self, pool: OligoPool, encoder: DNAEncoder):
        """Last `overlap` bases of oligo[i] should equal first `overlap` bases of oligo[i+1]."""
        data = os.urandom(500)
        master = encoder.encode_bytes(data)
        oligos = pool.fragment(master, encoder.start_bases)
        ov = pool.overlap
        for i in range(len(oligos) - 1):
            tail = oligos[i].payload[-ov:]
            head = oligos[i + 1].payload[:ov]
            assert tail == head, (
                f"Overlap mismatch between oligo {i} and {i+1}:\n"
                f"  tail={tail}\n  head={head}"
            )

    def test_only_acgt(self, pool: OligoPool, encoder: DNAEncoder):
        data = os.urandom(200)
        master = encoder.encode_bytes(data)
        oligos = pool.fragment(master, encoder.start_bases)
        for o in oligos:
            invalid = set(o.full_seq) - set("ACGT")
            assert not invalid, f"Non-ACGT chars in oligo {o.index}: {invalid}"

    def test_start_base_in_flags(self, pool: OligoPool, encoder: DNAEncoder):
        data = os.urandom(300)
        master = encoder.encode_bytes(data)
        oligos = pool.fragment(master, encoder.start_bases)
        p = pool.primer_len
        i = pool.index_len
        f = pool.FLAGS_LEN
        for o in oligos:
            flags_field = o.full_seq[p+i : p+i+f]
            recovered_sb = _FLAG_DEC[flags_field]
            assert recovered_sb == o.start_base

    def test_large_pool_index_capacity(self):
        """A pool with index_len=8 supports up to 65 536 oligos."""
        pool = OligoPool(oligo_len=150, overlap=5, primer_len=20, index_len=8)
        assert pool.info()["max_oligos"] == 65536

class TestAssemble:
    def test_ordered_reconstruction(self, pool: OligoPool, encoder: DNAEncoder):
        data = os.urandom(400)
        master = encoder.encode_bytes(data)
        oligos = pool.fragment(master, encoder.start_bases)
        rebuilt, _ = pool.assemble(oligos)
        assert rebuilt[:len(master)] == master

    def test_disordered_reconstruction(self, pool: OligoPool, encoder: DNAEncoder):
        data = os.urandom(400)
        master = encoder.encode_bytes(data)
        oligos = pool.fragment(master, encoder.start_bases)
        shuffled = oligos.copy()
        random.shuffle(shuffled)
        rebuilt, _ = pool.assemble(shuffled)
        assert rebuilt[:len(master)] == master

    def test_duplicate_oligos_consensus(self, pool: OligoPool, encoder: DNAEncoder):
        """Pool with duplicates should still reconstruct correctly."""
        data = os.urandom(300)
        master = encoder.encode_bytes(data)
        oligos = pool.fragment(master, encoder.start_bases)
        doubled = oligos + oligos
        random.shuffle(doubled)
        rebuilt, _ = pool.assemble(doubled)
        assert rebuilt[:len(master)] == master

    def test_start_bases_recovered(self, pool: OligoPool, encoder: DNAEncoder):
        data = os.urandom(200)
        master = encoder.encode_bytes(data)
        oligos = pool.fragment(master, encoder.start_bases)
        _, recovered_sbs = pool.assemble(oligos)
        for i, o in enumerate(oligos):
            assert recovered_sbs[i] == o.start_base

class TestFASTARoundTrip:
    def test_write_read_roundtrip(self, pool: OligoPool, encoder: DNAEncoder, tmp_path):
        data = os.urandom(300)
        master = encoder.encode_bytes(data)
        oligos = pool.fragment(master, encoder.start_bases)

        fasta_path = tmp_path / "pool.fasta"
        pool.write_fasta(oligos, fasta_path)

        loaded = pool.read_fasta(fasta_path)
        assert len(loaded) == len(oligos)

        for orig, loaded_o in zip(oligos, loaded):
            assert orig.index == loaded_o.index
            assert orig.payload == loaded_o.payload
            assert orig.start_base == loaded_o.start_base
            assert orig.full_seq == loaded_o.full_seq

    def test_fasta_format(self, pool: OligoPool, encoder: DNAEncoder, tmp_path):
        data = b"hello FASTA"
        master = encoder.encode_bytes(data)
        oligos = pool.fragment(master, encoder.start_bases)

        fasta_path = tmp_path / "test.fasta"
        pool.write_fasta(oligos, fasta_path)

        content = fasta_path.read_text()
        lines = content.strip().split("\n")
        headers = [l for l in lines if l.startswith(">")]
        assert len(headers) == len(oligos)
        for h in headers:
            assert "start=" in h

    def test_discard_corrupted_primer(self, pool: OligoPool, encoder: DNAEncoder, tmp_path):
        """FASTA read should still work even if header fields are minimal."""
        fasta_path = tmp_path / "minimal.fasta"
        fasta_path.write_text(
            ">oligo_000000\n"
            + pool.primer_fwd
            + _int_to_bases(0, pool.index_len)
            + _FLAG_ENC["A"]
            + "ACGT" * (pool.payload_len // 4)
            + pool.primer_rev_rc
            + "\n"
        )
        loaded = pool.read_fasta(fasta_path)
        assert len(loaded) == 1
        assert loaded[0].index == 0

class TestFullPipeline:
    def test_round_trip_small(self, pool: OligoPool, encoder: DNAEncoder):
        data = b"Integration test payload 12345"
        master = encoder.encode_bytes(data)
        oligos = pool.fragment(master, encoder.start_bases)
        rebuilt_master, rebuilt_sbs = pool.assemble(oligos)

        recovered = encoder.decode_sequence(rebuilt_master[:len(master)], encoder.start_bases)
        assert recovered == data

    def test_round_trip_1kb(self):
        enc = DNAEncoder(block_size=200)
        pl  = OligoPool(oligo_len=150, overlap=20, primer_len=20, index_len=8)
        data = os.urandom(1024)

        master = enc.encode_bytes(data)
        oligos = pl.fragment(master, enc.start_bases)

        random.shuffle(oligos)
        rebuilt_master, _ = pl.assemble(oligos)

        recovered = enc.decode_sequence(rebuilt_master[:len(master)], enc.start_bases)
        assert recovered == data

    def test_oligo_count_scales_with_data(self):
        enc  = DNAEncoder(block_size=200)
        pool = OligoPool(oligo_len=150, overlap=20, primer_len=20, index_len=8)

        data_small = os.urandom(100)
        data_large = os.urandom(1000)

        oligos_small = pool.fragment(enc.encode_bytes(data_small), enc.start_bases)
        oligos_large = pool.fragment(enc.encode_bytes(data_large), enc.start_bases)

        assert len(oligos_large) > len(oligos_small)

    @pytest.mark.slow
    def test_round_trip_50kb(self):
        enc  = DNAEncoder(block_size=200)
        pool = OligoPool(oligo_len=150, overlap=20, primer_len=20, index_len=8)
        data = os.urandom(50 * 1024)

        master = enc.encode_bytes(data)
        oligos = pool.fragment(master, enc.start_bases)
        random.shuffle(oligos)
        rebuilt_master, _ = pool.assemble(oligos)

        recovered = enc.decode_sequence(rebuilt_master[:len(master)], enc.start_bases)
        assert recovered == data
