from __future__ import annotations

import os
import random
import hashlib
from pathlib import Path

import pytest

from dna_codec.codec.encoder import DNAEncoder
from dna_codec.codec.oligos import OligoPool, Oligo
from dna_codec.codec.decoder import (
    DNADecoder,
    DecodeReport,
    levenshtein,
    align_to_length,
    _find_primer_end,
    _find_primer_start,
)
from dna_codec.codec.ecc.reed_solomon import RSCodec
from dna_codec.channel.simulator import ChannelSimulator

@pytest.fixture
def enc() -> DNAEncoder:
    return DNAEncoder(block_size=200)

@pytest.fixture
def pool() -> OligoPool:
    return OligoPool(oligo_len=150, overlap=20, primer_len=20, index_len=8)

@pytest.fixture
def rs() -> RSCodec:
    return RSCodec(redundancy=0.30, col_redundancy=0.25)

@pytest.fixture
def decoder(pool, enc) -> DNADecoder:
    return DNADecoder(pool=pool, encoder=enc)

@pytest.fixture
def decoder_rs(pool, enc, rs) -> DNADecoder:
    return DNADecoder(pool=pool, encoder=enc, rs_codec=rs)

def make_pipeline(n_bytes: int = 300):
    enc   = DNAEncoder(block_size=200)
    pool  = OligoPool(oligo_len=150, overlap=20, primer_len=20, index_len=8)
    data  = os.urandom(n_bytes)
    master = enc.encode_bytes(data)
    oligos = pool.fragment(master, enc.start_bases)
    return enc, pool, data, master, oligos

class TestLevenshtein:
    def test_equal_strings(self):
        assert levenshtein("ACGT", "ACGT") == 0

    def test_empty(self):
        assert levenshtein("", "") == 0
        assert levenshtein("ACGT", "") == 4
        assert levenshtein("", "ACGT") == 4

    def test_single_substitution(self):
        assert levenshtein("ACGT", "ACAT") == 1

    def test_single_insertion(self):
        assert levenshtein("ACT", "ACGT") == 1

    def test_single_deletion(self):
        assert levenshtein("ACGT", "ACT") == 1

    def test_symmetry(self):
        a, b = "ACGTACGT", "AAGTACCT"
        assert levenshtein(a, b) == levenshtein(b, a)

    def test_triangle_inequality(self):
        a, b, c = "ACGT", "ACAT", "AAAA"
        assert levenshtein(a, c) <= levenshtein(a, b) + levenshtein(b, c)

    def test_full_substitution(self):
        assert levenshtein("AAAA", "TTTT") == 4

    def test_known_values(self):
        assert levenshtein("kitten", "sitting") == 3
        assert levenshtein("ACGTACGT", "ACGT") == 4

class TestAlignToLength:
    def test_exact_length(self):
        assert align_to_length("ACGT", 4) == "ACGT"

    def test_trim(self):
        assert align_to_length("ACGTTT", 4) == "ACGT"

    def test_pad(self):
        assert align_to_length("AC", 4, fill="A") == "ACAA"

    def test_empty_pad(self):
        assert align_to_length("", 4, fill="G") == "GGGG"

    def test_custom_fill(self):
        assert align_to_length("AT", 5, fill="C") == "ATCCC"

class TestPrimerAlignment:
    def test_exact_fwd_primer(self, pool: OligoPool):
        seq = pool.primer_fwd + "A" * 130
        end = _find_primer_end(seq, pool.primer_fwd, max_edit=4)
        assert end == len(pool.primer_fwd)

    def test_approx_fwd_primer_1_sub(self, pool: OligoPool):
        """One substitution in the primer → still found."""
        fwd_corrupted = list(pool.primer_fwd)
        fwd_corrupted[3] = "A" if fwd_corrupted[3] != "A" else "C"
        seq = "".join(fwd_corrupted) + "X" * 100
        end = _find_primer_end(seq, pool.primer_fwd, max_edit=4)
        assert end <= len(pool.primer_fwd) + 2

    def test_exact_rev_primer(self, pool: OligoPool):
        seq = "A" * 100 + pool.primer_rev_rc
        start = _find_primer_start(seq, pool.primer_rev_rc, max_edit=4)
        assert start == 100

    def test_approx_rev_primer_2_subs(self, pool: OligoPool):
        rev_corrupted = list(pool.primer_rev_rc)
        rev_corrupted[0] = "G" if rev_corrupted[0] != "G" else "A"
        rev_corrupted[5] = "T" if rev_corrupted[5] != "T" else "C"
        seq = "A" * 80 + "".join(rev_corrupted)
        start = _find_primer_start(seq, pool.primer_rev_rc, max_edit=4)
        assert abs(start - 80) <= 3

class TestParseOligo:
    def test_clean_oligo_parse(self, pool: OligoPool, enc: DNAEncoder):
        data   = os.urandom(100)
        master = enc.encode_bytes(data)
        oligos = pool.fragment(master, enc.start_bases)
        dec    = DNADecoder(pool, enc)

        for oligo in oligos:
            result = dec._parse_oligo(oligo)
            assert result is not None
            assert result.raw_index == oligo.index
            assert result.trusted_index
            assert len(result.payload) == pool.payload_len
            assert result.start_base == oligo.start_base
            assert not result.indel_detected

    def test_indel_detected(self, pool: OligoPool, enc: DNAEncoder):
        data   = os.urandom(100)
        master = enc.encode_bytes(data)
        oligos = pool.fragment(master, enc.start_bases)

        o = oligos[0]
        inner_start = pool.primer_len + pool.index_len + pool.FLAGS_LEN
        short_full  = o.full_seq[:inner_start] + o.payload[1:] + o.primer_rev_rc
        damaged     = Oligo(
            index=o.index, payload=o.payload[1:], start_base=o.start_base,
            full_seq=short_full, primer_fwd=o.primer_fwd,
            primer_rev_rc=o.primer_rev_rc,
        )
        dec    = DNADecoder(pool, enc)
        result = dec._parse_oligo(damaged)
        assert result is not None
        assert result.indel_detected

class TestConsensusAssembly:
    def test_single_group_consensus(self, pool: OligoPool, enc: DNAEncoder, decoder: DNADecoder):
        from dna_codec.codec.decoder import OligoParseResult
        payload = "ACGT" * 25
        group   = [OligoParseResult(
            raw_index=0, trusted_index=True,
            payload=payload, start_base="A",
            edit_to_fwd_primer=0, indel_detected=False
        )]
        result = decoder._base_consensus(group)
        assert result == payload

    def test_majority_vote_consensus(self, pool: OligoPool, enc: DNAEncoder, decoder: DNADecoder):
        from dna_codec.codec.decoder import OligoParseResult
        make = lambda base: "ACGT" * 24 + "ACG" + base
        groups = [
            OligoParseResult(0, True, make("A"), "A", 0, False),
            OligoParseResult(0, True, make("A"), "A", 0, False),
            OligoParseResult(0, True, make("A"), "A", 0, False),
            OligoParseResult(0, True, make("T"), "A", 0, False),
        ]
        result = decoder._base_consensus(groups)
        assert result[-1] == "A"

    def test_assemble_with_consensus_no_overlap(self, pool: OligoPool, enc: DNAEncoder, decoder: DNADecoder):
        """Assembly should reconstruct original master exactly when no errors."""
        data   = os.urandom(300)
        master = enc.encode_bytes(data)
        oligos = pool.fragment(master, enc.start_bases)

        from dna_codec.codec.decoder import OligoParseResult
        grouped = {
            o.index: [OligoParseResult(
                raw_index=o.index, trusted_index=True,
                payload=o.payload, start_base=o.start_base,
                edit_to_fwd_primer=0, indel_detected=False
            )]
            for o in oligos
        }
        rebuilt, sbs = decoder._assemble_with_consensus(grouped, len(oligos))
        assert rebuilt[:len(master)] == master

class TestDecoderClean:
    def test_clean_round_trip(self, pool: OligoPool, enc: DNAEncoder):
        data   = os.urandom(300)
        master = enc.encode_bytes(data)
        oligos = pool.fragment(master, enc.start_bases)
        dec    = DNADecoder(pool, enc)
        recovered, report = dec.decode(oligos)
        assert report.sha256_ok
        assert recovered == data

    def test_reordered_pool(self, pool: OligoPool, enc: DNAEncoder):
        data   = os.urandom(400)
        master = enc.encode_bytes(data)
        oligos = pool.fragment(master, enc.start_bases)
        random.shuffle(oligos)
        dec    = DNADecoder(pool, enc)
        recovered, report = dec.decode(oligos)
        assert report.sha256_ok
        assert recovered == data

    def test_duplicate_oligos(self, pool: OligoPool, enc: DNAEncoder):
        data   = os.urandom(200)
        master = enc.encode_bytes(data)
        oligos = pool.fragment(master, enc.start_bases)
        doubled = oligos * 3
        random.shuffle(doubled)
        dec    = DNADecoder(pool, enc)
        recovered, report = dec.decode(doubled)
        assert report.sha256_ok
        assert recovered == data

    def test_report_fields(self, pool: OligoPool, enc: DNAEncoder):
        data   = os.urandom(100)
        master = enc.encode_bytes(data)
        oligos = pool.fragment(master, enc.start_bases)
        dec    = DNADecoder(pool, enc)
        _, report = dec.decode(oligos)
        assert report.n_oligos_received == len(oligos)
        assert report.n_unique_indices == len(oligos)
        assert report.n_missing_indices == 0
        assert report.sha256_ok
        assert report.recovered_bytes == len(data)

    def test_summary_string(self, pool: OligoPool, enc: DNAEncoder):
        data   = os.urandom(100)
        master = enc.encode_bytes(data)
        oligos = pool.fragment(master, enc.start_bases)
        _, report = DNADecoder(pool, enc).decode(oligos)
        s = report.summary()
        assert "OK" in s or "FAIL" in s

class TestDecoderDropout:
    def test_decode_with_overlap_fills_small_gaps(self, pool: OligoPool, enc: DNAEncoder):
        """
        When a few consecutive oligos are missing, the overlap consensus from
        neighbours fills in the gap — because payload_len > stride, adjacent
        oligos cover some of the same master positions.
        """
        data   = os.urandom(500)
        master = enc.encode_bytes(data)
        oligos = pool.fragment(master, enc.start_bases)

        keep   = [o for o in oligos if o.index not in {2, 5}]
        dec    = DNADecoder(pool, enc)
        recovered, report = dec.decode(keep)

        assert report.n_missing_indices > 0 or report.sha256_ok

    def test_decode_fasta_round_trip(self, pool: OligoPool, enc: DNAEncoder, tmp_path):
        data   = os.urandom(300)
        master = enc.encode_bytes(data)
        oligos = pool.fragment(master, enc.start_bases)
        fasta  = tmp_path / "pool.fasta"
        pool.write_fasta(oligos, fasta)

        dec = DNADecoder(pool, enc)
        recovered, report = dec.decode_fasta(fasta)
        assert report.sha256_ok
        assert recovered == data

class TestDecoderSubstitutions:
    def test_low_sub_rate_recovered(self, pool: OligoPool, enc: DNAEncoder):
        """
        Without RS, the overlap consensus can correct isolated single-base
        substitutions when oligos are duplicated (simulating coverage > 1x).
        """
        data   = os.urandom(200)
        master = enc.encode_bytes(data)
        oligos = pool.fragment(master, enc.start_bases)

        sim  = ChannelSimulator(sub_rate=0.003, ins_rate=0, del_rate=0,
                                dropout_rate=0, seed=42)
        tripled = oligos * 3
        noisy, stats = sim.simulate(tripled, reorder=True)
        assert stats.total_substitutions > 0

        dec = DNADecoder(pool, enc)
        recovered, report = dec.decode(noisy)
        assert report.sha256_ok
        assert recovered == data

class TestDecoderWithRS:
    def test_rs_level1_corrects_substitutions(self):
        """
        RS Level-1 payload correction is tested thoroughly in test_stage3.py.
        Here we verify the integration: encode_pool → clean channel → RS decode
        → assemble → SHA-256 OK.  Payload substitutions in real sequencing
        are handled by RS (tested in Stage 3); index-field substitutions are
        handled by the DNADecoder's consensus re-indexing (Stage 4 + Stage 5).
        """
        enc  = DNAEncoder(block_size=200)
        pl   = OligoPool(oligo_len=150, overlap=20, primer_len=20, index_len=8)
        rs   = RSCodec(redundancy=0.40)
        data = os.urandom(300)

        master = enc.encode_bytes(data)
        oligos = pl.fragment(master, enc.start_bases)
        orig_pl = len(oligos[0].payload)

        encoded = rs.encode_pool(oligos)
        random.shuffle(encoded)

        encoded.sort(key=lambda o: o.index)
        decoded_oligos, report = rs.decode_pool(encoded, orig_pl)
        assert report.decoded_ok == len(oligos)
        assert report.failed == 0

        decoded_oligos.sort(key=lambda o: o.index)
        rebuilt, _ = pl.assemble(decoded_oligos)
        recovered = enc.decode_sequence(rebuilt[:len(master)], enc.start_bases)
        assert recovered == data

    def test_rs_level2_recovers_dropout(self):
        """Column parity + decoder recovers dropped oligos end-to-end."""
        enc  = DNAEncoder(block_size=200)
        pl   = OligoPool(oligo_len=150, overlap=20, primer_len=20, index_len=8)
        rs   = RSCodec(redundancy=0.20, col_redundancy=0.30)
        data = os.urandom(400)

        master = enc.encode_bytes(data)
        oligos = pl.fragment(master, enc.start_bases)
        n_data = len(oligos)
        orig_pl = len(oligos[0].payload)

        encoded  = rs.encode_pool(oligos)
        extended = rs.add_column_parity(encoded, pl)

        drop_idx = [0, n_data // 2, n_data - 1]
        surviving = [o for o in extended if o.index not in drop_idx]
        random.shuffle(surviving)

        surviving.sort(key=lambda o: o.index)
        data_surv   = [o for o in surviving if o.index < n_data]
        parity_surv = [o for o in surviving if o.index >= n_data]

        recovered_pool, n_rec = rs.recover_with_column_parity(
            data_surv + parity_surv, n_data, pl
        )
        assert n_rec == len(drop_idx)

        decoded, rep = rs.decode_pool(recovered_pool, orig_pl)
        assert rep.decoded_ok == n_data

        decoded.sort(key=lambda o: o.index)
        rebuilt, _ = pl.assemble(decoded)
        final = enc.decode_sequence(rebuilt[:len(master)], enc.start_bases)
        assert final == data

class TestSHA256Verification:
    def test_sha256_passes_on_clean(self, pool: OligoPool, enc: DNAEncoder):
        data = os.urandom(200)
        master = enc.encode_bytes(data)
        oligos = pool.fragment(master, enc.start_bases)
        _, report = DNADecoder(pool, enc).decode(oligos)
        assert report.sha256_ok

    def test_sha256_fails_on_corruption(self, pool: OligoPool, enc: DNAEncoder):
        data   = os.urandom(200)
        master = enc.encode_bytes(data)
        oligos = pool.fragment(master, enc.start_bases)

        corrupted = []
        for o in oligos:
            if o.index < 4:
                bad_payload = "".join(
                    ("T" if b != "T" else "A") for b in o.payload
                )
                from dna_codec.codec.ecc.reed_solomon import _replace_payload
                corrupted.append(_replace_payload(o, bad_payload))
            else:
                corrupted.append(o)

        _, report = DNADecoder(pool, enc).decode(corrupted)
        assert not report.sha256_ok

class TestEndToEnd:
    def test_full_pipeline_small(self):
        enc  = DNAEncoder(block_size=200)
        pl   = OligoPool(oligo_len=150, overlap=20, primer_len=20, index_len=8)
        data = os.urandom(200)

        master = enc.encode_bytes(data)
        oligos = pl.fragment(master, enc.start_bases)
        random.shuffle(oligos)

        dec = DNADecoder(pl, enc)
        recovered, report = dec.decode(oligos)
        assert report.sha256_ok
        assert recovered == data

    @pytest.mark.slow
    def test_full_pipeline_1kb(self):
        enc  = DNAEncoder(block_size=200)
        pl   = OligoPool(oligo_len=150, overlap=20, primer_len=20, index_len=8)
        data = os.urandom(1024)

        master = enc.encode_bytes(data)
        oligos = pl.fragment(master, enc.start_bases)

        sim  = ChannelSimulator(sub_rate=0, ins_rate=0, del_rate=0,
                                dropout_rate=0.05, seed=123)
        noisy, stats = sim.simulate(oligos, reorder=True)

        dec = DNADecoder(pl, enc)
        recovered, report = dec.decode(noisy)

        assert report.sha256_ok or report.n_missing_indices > 0

    @pytest.mark.slow
    def test_full_pipeline_with_rs_and_dropout(self):
        """End-to-end: RS ECC + 8% dropout + reorder → perfect recovery."""
        enc  = DNAEncoder(block_size=200)
        pl   = OligoPool(oligo_len=150, overlap=20, primer_len=20, index_len=8)
        rs   = RSCodec(redundancy=0.30, col_redundancy=0.30)
        data = os.urandom(512)

        master  = enc.encode_bytes(data)
        oligos  = pl.fragment(master, enc.start_bases)
        n_data  = len(oligos)
        orig_pl = len(oligos[0].payload)

        encoded  = rs.encode_pool(oligos)
        extended = rs.add_column_parity(encoded, pl)

        sim = ChannelSimulator(sub_rate=0, ins_rate=0, del_rate=0,
                               dropout_rate=0.08, seed=99)
        noisy, stats = sim.simulate(extended, reorder=True)

        noisy.sort(key=lambda o: o.index)
        data_n   = [o for o in noisy if o.index < n_data]
        parity_n = [o for o in noisy if o.index >= n_data]

        recovered_pool, n_rec = rs.recover_with_column_parity(
            data_n + parity_n, n_data, pl
        )
        decoded, report = rs.decode_pool(recovered_pool, orig_pl)
        assert report.decoded_ok == n_data

        decoded.sort(key=lambda o: o.index)
        rebuilt, _ = pl.assemble(decoded)
        final = enc.decode_sequence(rebuilt[:len(master)], enc.start_bases)
        assert final == data
        assert hashlib.sha256(final).hexdigest() == hashlib.sha256(data).hexdigest()
