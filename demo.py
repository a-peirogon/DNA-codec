#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import random
import textwrap

from dna_codec.codec.encoder import DNAEncoder
from dna_codec.codec.oligos import OligoPool
from dna_codec.codec.decoder import DNADecoder
from dna_codec.codec.ecc.reed_solomon import RSCodec
from dna_codec.channel.simulator import ChannelSimulator

MESSAGE = (
    b"Hello, DNA data storage world! This is an end-to-end demonstration "
    b"of the full pipeline: encoding, oligo synthesis, noisy channel "
    b"simulation, Reed-Solomon error correction, and decoding back to "
    b"the original message."
)

def banner(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)

def scenario_with_ecc(message: bytes, seed: int = 1) -> None:
    """Realistic scenario: oligo dropout, corrected via Reed-Solomon."""
    banner("SCENARIO 1 — Oligo dropout, corrected with Reed-Solomon ECC")

    enc = DNAEncoder(block_size=200)
    pool = OligoPool(oligo_len=150, overlap=20, primer_len=20, index_len=8)
    rs = RSCodec(redundancy=0.30, col_redundancy=0.30)

    master = enc.encode_bytes(message)
    oligos = pool.fragment(master, enc.start_bases)
    n_data = len(oligos)
    orig_payload_len = len(oligos[0].payload)
    print(f"Message size:        {len(message)} bytes")
    print(f"DNA sequence length: {len(master)} bases")
    print(f"Data oligos:         {n_data}")

    encoded = rs.encode_pool(oligos)
    extended = rs.add_column_parity(encoded, pool)
    print(f"Oligos with ECC (data + parity): {len(extended)}")

    sim = ChannelSimulator(
        sub_rate=0.0, ins_rate=0.0, del_rate=0.0,
        dropout_rate=0.08, seed=seed,
    )
    noisy, stats = sim.simulate(extended, reorder=True)
    print(f"Oligos lost in channel: {stats.n_dropped} / {stats.n_oligos_in} "
          f"({stats.dropout_rate:.0%} dropout rate)")

    noisy.sort(key=lambda o: o.index)
    data_n = [o for o in noisy if o.index < n_data]
    parity_n = [o for o in noisy if o.index >= n_data]

    recovered_pool, n_rec = rs.recover_with_column_parity(
        data_n + parity_n, n_data, pool
    )
    decoded, report = rs.decode_pool(recovered_pool, orig_payload_len)
    print(f"RS-decoded oligos ok:    {report.decoded_ok} / {n_data}")

    decoded.sort(key=lambda o: o.index)
    rebuilt, _ = pool.assemble(decoded)
    final = enc.decode_sequence(rebuilt[: len(master)], enc.start_bases)

    ok = final == message
    sha_ok = hashlib.sha256(final).hexdigest() == hashlib.sha256(message).hexdigest()
    print()
    print(f"Recovered message == original: {ok}")
    print(f"SHA-256 verified:               {sha_ok}")
    print()
    print("Recovered text:")
    print(textwrap.indent(textwrap.fill(final.decode(), 74), "  "))

def scenario_without_ecc(message: bytes, seed: int = 11) -> None:
    """Lightweight scenario: no Reed-Solomon layer, just fragment + a
    clean/reordered channel, reassembled via primer alignment and overlap
    consensus. This shows the decoder's baseline path (index recovery,
    consensus assembly, SHA-256 check) without RS involved.

    Note: payload-level bit errors (substitutions/indels) are corrected by
    the RS layer (see Scenario 1), not by this baseline path alone -- that
    correction is by design, not a bug. See README, 'How decoding works'.
    """
    banner("SCENARIO 2 -- No ECC, reordered but otherwise clean channel")

    enc = DNAEncoder(block_size=200)
    pool = OligoPool(oligo_len=150, overlap=20, primer_len=20, index_len=8)

    master = enc.encode_bytes(message)
    oligos = pool.fragment(master, enc.start_bases)

    sim = ChannelSimulator(
        sub_rate=0.0, ins_rate=0.0, del_rate=0.0,
        dropout_rate=0.0, seed=seed,
    )
    noisy, stats = sim.simulate(oligos, reorder=True)
    print(f"Oligos reordered by channel simulator: {stats.n_oligos_out}")

    dec = DNADecoder(pool, enc)
    recovered, report = dec.decode(noisy)

    print(f"SHA-256 verified: {report.sha256_ok}")
    print(f"Recovered message == original: {recovered == message}")

def main() -> None:
    random.seed(42)
    scenario_with_ecc(MESSAGE)
    scenario_without_ecc(MESSAGE)
    banner("Done")

if __name__ == "__main__":
    main()
