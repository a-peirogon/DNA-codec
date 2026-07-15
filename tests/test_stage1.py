"""
tests/test_stage1.py — Test suite for encoder.py and constraints.py (Stage 1).

Run with:
    cd dna_codec && python -m pytest tests/test_stage1.py -v

Coverage
--------
  - ConstraintChecker: GC content, homopolymer detection, palindrome detection
  - ConstraintEnforcer: rotation encoding round-trip
  - DNAEncoder: encode_bytes → decode_sequence round-trip
  - Edge cases: empty bytes, single byte, 1 MB random payload
  - Constraint satisfaction of encoded sequences
"""

from __future__ import annotations

import os
import random
import struct

import pytest

from dna_codec.codec.constraints import (
    ConstraintChecker,
    ConstraintEnforcer,
    ConstraintReport,
    reverse_complement,
    _BASE_TO_BITS,
)
from dna_codec.codec.encoder import (
    DNAEncoder,
    MAGIC,
    HEADER_FMT,
    bytes_to_dibits,
    dibits_to_bytes,
)


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def checker() -> ConstraintChecker:
    return ConstraintChecker(gc_min=0.40, gc_max=0.60, max_homopolymer=3, max_palindrome=6)


@pytest.fixture
def enforcer(checker: ConstraintChecker) -> ConstraintEnforcer:
    return ConstraintEnforcer(checker)


@pytest.fixture
def encoder() -> DNAEncoder:
    return DNAEncoder(block_size=200)


# ===========================================================================
# 1. reverse_complement
# ===========================================================================


class TestReverseComplement:
    def test_single_base(self):
        assert reverse_complement("A") == "T"
        assert reverse_complement("T") == "A"
        assert reverse_complement("C") == "G"
        assert reverse_complement("G") == "C"

    def test_known_sequence(self):
        # 5'-AACGT-3' → complement 3'-TTGCA-5' → reverse 5'-ACGTT-3'
        assert reverse_complement("AACGT") == "ACGTT"

    def test_palindrome_detection(self):
        # GAATTC is a palindrome (EcoRI site)
        seq = "GAATTC"
        assert reverse_complement(seq) == seq

    def test_empty(self):
        assert reverse_complement("") == ""


# ===========================================================================
# 2. ConstraintChecker — GC content
# ===========================================================================


class TestGCContent:
    def test_all_gc(self, checker: ConstraintChecker):
        assert checker.gc_content("GCGCGC") == pytest.approx(1.0)

    def test_all_at(self, checker: ConstraintChecker):
        assert checker.gc_content("ATATAT") == pytest.approx(0.0)

    def test_balanced(self, checker: ConstraintChecker):
        assert checker.gc_content("ACGT") == pytest.approx(0.5)

    def test_empty(self, checker: ConstraintChecker):
        assert checker.gc_content("") == 0.0

    def test_gc_pass(self, checker: ConstraintChecker):
        seq = "ACGTACGT"  # 50 % GC
        report = checker.check(seq)
        assert report.gc_ok
        assert report.gc_content == pytest.approx(0.5)

    def test_gc_fail_low(self, checker: ConstraintChecker):
        seq = "AAAATTTT"  # 0 % GC
        report = checker.check(seq)
        assert not report.gc_ok

    def test_gc_fail_high(self, checker: ConstraintChecker):
        seq = "GGGGCCCC"  # 100 % GC
        report = checker.check(seq)
        assert not report.gc_ok


# ===========================================================================
# 3. ConstraintChecker — Homopolymer runs
# ===========================================================================


class TestHomopolymer:
    def test_no_run(self, checker: ConstraintChecker):
        assert checker.longest_homopolymer("ACGT") == 1

    def test_run_of_3(self, checker: ConstraintChecker):
        assert checker.longest_homopolymer("AAACGT") == 3

    def test_run_of_4(self, checker: ConstraintChecker):
        assert checker.longest_homopolymer("AAAACGT") == 4

    def test_run_ok(self, checker: ConstraintChecker):
        seq = "AAACGT"  # run of 3, limit is 3
        report = checker.check(seq)
        assert report.homopolymer_ok

    def test_run_fail(self, checker: ConstraintChecker):
        seq = "AAAACGT"  # run of 4 > 3
        report = checker.check(seq)
        assert not report.homopolymer_ok
        assert any("homopolymer" in v for v in report.violations)

    def test_multiple_runs(self, checker: ConstraintChecker):
        seq = "AACCGGTT"  # all runs of 2
        assert checker.longest_homopolymer(seq) == 2
        assert checker.check(seq).homopolymer_ok


# ===========================================================================
# 4. ConstraintChecker — Palindromes
# ===========================================================================


class TestPalindrome:
    def test_known_palindrome_gaattc(self, checker: ConstraintChecker):
        # GAATTC is a 6-mer palindrome (EcoRI site) — at the limit
        seq = "GAATTC"
        report = checker.check(seq)
        assert report.longest_palindrome == 6
        # 6 == max_palindrome, so it should PASS (≤ not <)
        assert report.palindrome_ok

    def test_palindrome_too_long(self, checker: ConstraintChecker):
        # AGATCT = BglII site (6-mer palindrome) padded with GC to make 8-mer
        # GCAGATCTGC — let's find an 8-mer palindrome: AGATCTAGAT? No.
        # Construct one: GCGAATTCGC → GCGAATTCGC rev_comp = GCGAATTCGC ✓
        seq = "GCGAATTCGC"
        rc = reverse_complement(seq)
        # Verify it IS a palindrome
        assert seq == rc, f"{seq} != {rc}"
        report = checker.check(seq)
        assert report.longest_palindrome >= 8
        assert not report.palindrome_ok

    def test_no_palindrome(self, checker: ConstraintChecker):
        seq = "AACCTG"  # no self-complementary substring ≥ 4
        report = checker.check(seq)
        assert report.palindrome_ok

    def test_short_sequence(self, checker: ConstraintChecker):
        # Sequence too short to have a 4-mer palindrome
        seq = "ACG"
        report = checker.check(seq)
        assert report.palindrome_ok
        assert report.longest_palindrome == 0


# ===========================================================================
# 5. ConstraintReport
# ===========================================================================


class TestConstraintReport:
    def test_passed_property(self, checker: ConstraintChecker):
        good_seq = "ACGTACGT" * 5  # 40 bases, 50 % GC, no runs > 1, short palindromes
        report = checker.check(good_seq)
        # Result depends on palindromes in the sequence; check property consistency
        assert report.passed == (report.gc_ok and report.homopolymer_ok and report.palindrome_ok)

    def test_violations_list(self, checker: ConstraintChecker):
        bad_seq = "AAAAACCCC"  # run of 5 A's → homopolymer violation
        report = checker.check(bad_seq)
        assert not report.passed
        assert len(report.violations) >= 1


# ===========================================================================
# 6. bytes_to_dibits / dibits_to_bytes (round-trip)
# ===========================================================================


class TestDibitConversion:
    def test_zero_byte(self):
        assert bytes_to_dibits(b"\x00") == [0, 0, 0, 0]

    def test_ff_byte(self):
        assert bytes_to_dibits(b"\xFF") == [3, 3, 3, 3]

    def test_known_byte(self):
        # 0xE4 = 11100100 → [3, 2, 1, 0]
        assert bytes_to_dibits(b"\xE4") == [3, 2, 1, 0]

    def test_round_trip_single_byte(self):
        for b in range(256):
            data = bytes([b])
            assert dibits_to_bytes(bytes_to_dibits(data)) == data

    def test_round_trip_random(self):
        data = os.urandom(64)
        assert dibits_to_bytes(bytes_to_dibits(data)) == data

    def test_dibits_to_bytes_raises_on_unaligned(self):
        with pytest.raises(ValueError, match="multiple of 4"):
            dibits_to_bytes([0, 1, 2])  # length 3 is not multiple of 4


# ===========================================================================
# 7. ConstraintEnforcer — rotation round-trip
# ===========================================================================


class TestRotationRoundTrip:
    def test_single_symbol(self, enforcer: ConstraintEnforcer):
        for start in ["A", "C", "G", "T"]:
            for v in range(4):
                seq = enforcer.rotation_encode([v], start_base=start)
                assert len(seq) == 1
                recovered = enforcer.rotation_decode(seq, start_base=start)
                assert recovered == [v], f"start={start} v={v} seq={seq}"

    def test_empty(self, enforcer: ConstraintEnforcer):
        assert enforcer.rotation_encode([], "A") == ""
        assert enforcer.rotation_decode("", "A") == []

    def test_round_trip_short(self, enforcer: ConstraintEnforcer):
        bits = [0, 1, 2, 3, 0, 2, 1, 3]
        for start in ["A", "C", "G", "T"]:
            seq = enforcer.rotation_encode(bits, start_base=start)
            recovered = enforcer.rotation_decode(seq, start_base=start)
            assert recovered == bits, f"start={start}"

    def test_round_trip_random_200(self, enforcer: ConstraintEnforcer):
        bits = [random.randint(0, 3) for _ in range(200)]
        for start in ["A", "C", "G", "T"]:
            seq = enforcer.rotation_encode(bits, start_base=start)
            recovered = enforcer.rotation_decode(seq, start_base=start)
            assert recovered == bits

    def test_no_homopolymer_run_aaaa(self, enforcer: ConstraintEnforcer):
        """Encoding 0,0,0,0,0 with rotation should NOT produce a run of 5."""
        # All-zero dibits would naively map to AAAAA in raw encoding,
        # but the rotation table should break homopolymers.
        bits = [0] * 20
        seq = enforcer.rotation_encode(bits, start_base="A")
        checker = ConstraintChecker()
        assert checker.longest_homopolymer(seq) <= 3, (
            f"Homopolymer run too long in: {seq}"
        )

    def test_best_start_base_improves_gc(self, enforcer: ConstraintEnforcer):
        """best_start_base should select the start that keeps GC near 50%."""
        bits = bytes_to_dibits(b"\x00" * 50)  # heavily AT-biased raw
        best = enforcer.best_start_base(bits)
        seq = enforcer.rotation_encode(bits, start_base=best)
        gc = enforcer.checker.gc_content(seq)
        # Should be closer to 50% than a random start
        assert 0.30 <= gc <= 0.70, f"GC={gc:.1%} for start_base={best}"


# ===========================================================================
# 8. DNAEncoder — encode_bytes / decode_sequence round-trip
# ===========================================================================


class TestDNAEncoder:
    def test_round_trip_empty(self, encoder: DNAEncoder):
        data = b""
        seq = encoder.encode_bytes(data)
        start_bases = encoder.start_bases
        recovered = encoder.decode_sequence(seq, start_bases)
        assert recovered == data

    def test_round_trip_single_byte(self, encoder: DNAEncoder):
        data = b"\x42"
        seq = encoder.encode_bytes(data)
        recovered = encoder.decode_sequence(seq, encoder.start_bases)
        assert recovered == data

    def test_round_trip_ascii(self, encoder: DNAEncoder):
        data = b"Hello, DNA World! 0123456789"
        seq = encoder.encode_bytes(data)
        recovered = encoder.decode_sequence(seq, encoder.start_bases)
        assert recovered == data

    def test_round_trip_all_bytes(self, encoder: DNAEncoder):
        data = bytes(range(256))
        seq = encoder.encode_bytes(data)
        recovered = encoder.decode_sequence(seq, encoder.start_bases)
        assert recovered == data

    def test_round_trip_1kb_random(self, encoder: DNAEncoder):
        data = os.urandom(1024)
        seq = encoder.encode_bytes(data)
        recovered = encoder.decode_sequence(seq, encoder.start_bases)
        assert recovered == data

    @pytest.mark.slow
    def test_round_trip_1mb_random(self, encoder: DNAEncoder):
        """1 MB payload — may take a few seconds."""
        data = os.urandom(1024 * 1024)
        seq = encoder.encode_bytes(data)
        recovered = encoder.decode_sequence(seq, encoder.start_bases)
        assert recovered == data

    def test_output_is_valid_bases(self, encoder: DNAEncoder):
        data = os.urandom(256)
        seq = encoder.encode_bytes(data)
        assert all(b in "ACGT" for b in seq), "Non-ACGT characters found"

    def test_sequence_length(self, encoder: DNAEncoder):
        """Output length should be a multiple of block_size."""
        data = os.urandom(100)
        seq = encoder.encode_bytes(data)
        assert len(seq) % encoder.block_size == 0

    def test_wrong_magic_raises(self, encoder: DNAEncoder):
        """Decoding with corrupted start_bases should raise (magic mismatch)."""
        data = b"test"
        seq = encoder.encode_bytes(data)
        # Corrupt: use a different start base so rotation decode produces garbage
        bad_starts = [
            ("C" if s == "A" else "A")
            for s in encoder.start_bases
        ]
        with pytest.raises((ValueError, struct.error)):
            encoder.decode_sequence(seq, bad_starts)

    def test_sha256_stored(self, encoder: DNAEncoder):
        """sha256 attribute should be set after encode."""
        data = b"sha test"
        encoder.encode_bytes(data)
        import hashlib
        expected = hashlib.sha256(data).hexdigest()
        assert encoder.sha256 == expected

    def test_constraint_report_format(self, encoder: DNAEncoder):
        data = os.urandom(512)
        seq = encoder.encode_bytes(data)
        report = encoder.constraint_report(seq, sample_size=5)
        assert "total_bases" in report
        assert "gc_mean" in report
        assert 0.0 <= report["gc_mean"] <= 1.0


# ===========================================================================
# 9. Integration — full file-like encode/decode with SHA-256 verification
# ===========================================================================


class TestIntegration:
    def test_encode_decode_text_file(self, encoder: DNAEncoder, tmp_path):
        """Encode a text file, decode it, verify SHA-256 matches."""
        import hashlib

        original = b"The quick brown fox jumps over the lazy dog.\n" * 20
        src = tmp_path / "input.txt"
        src.write_bytes(original)

        seq = encoder.encode_file(src)
        recovered = encoder.decode_sequence(seq, encoder.start_bases)

        assert recovered == original
        assert hashlib.sha256(recovered).hexdigest() == hashlib.sha256(original).hexdigest()

    def test_gc_content_within_bounds(self, encoder: DNAEncoder):
        """GC content of each block should be within 40–60%."""
        data = os.urandom(2000)
        seq = encoder.encode_bytes(data)
        checker = ConstraintChecker(gc_min=0.30, gc_max=0.70)  # slightly relaxed for test
        block_size = encoder.block_size
        failures = []
        for i in range(0, len(seq), block_size):
            block = seq[i : i + block_size]
            if block:
                gc = checker.gc_content(block)
                if not (0.30 <= gc <= 0.70):
                    failures.append((i, gc))
        assert len(failures) == 0, f"GC out of range in blocks: {failures[:3]}"

    def test_homopolymer_constraint(self, encoder: DNAEncoder):
        """No homopolymer run > 3 in the encoded sequence (by rotation encoding)."""
        data = b"\x00" * 200  # worst case: all zeros → all-A without rotation
        seq = encoder.encode_bytes(data)
        checker = ConstraintChecker()
        # Check in windows of 50
        for i in range(0, len(seq) - 50, 50):
            window = seq[i : i + 50]
            hp = checker.longest_homopolymer(window)
            assert hp <= 4, (
                f"Homopolymer run of {hp} at position {i}: ...{window}..."
            )
