#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import sys

from dna_codec.codec.encoder import DNAEncoder
from dna_codec.codec.oligos import OligoPool
from dna_codec.codec.ecc.reed_solomon import RSCodec
from dna_codec.channel.simulator import ChannelSimulator

def encode_decode_image(
    image_path: str,
    out_path: str | None = None,
    dropout_rate: float = 0.10,
    sub_rate: float = 0.0,
    ins_rate: float = 0.0,
    del_rate: float = 0.0,
    seed: int = 7,
    oligo_len: int = 150,
    overlap: int = 20,
    primer_len: int = 20,
    index_len: int = 8,
    redundancy: float = 0.30,
    col_redundancy: float = 0.20,
) -> bool:
    """Codifica image_path a DNA, simula un canal ruidoso, decodifica de
    vuelta y compara con el original. Devuelve True si coincide byte a
    byte (SHA-256 verificado)."""

    with open(image_path, "rb") as f:
        image_bytes = f.read()

    print(f"Imagen de entrada: {image_path} ({len(image_bytes)} bytes)")

    enc = DNAEncoder(block_size=200)
    pool = OligoPool(
        oligo_len=oligo_len, overlap=overlap,
        primer_len=primer_len, index_len=index_len,
    )
    rs = RSCodec(redundancy=redundancy, col_redundancy=col_redundancy)

    master = enc.encode_bytes(image_bytes)
    oligos = pool.fragment(master, enc.start_bases)
    n_data = len(oligos)
    orig_payload_len = len(oligos[0].payload)
    print(f"Secuencia DNA: {len(master)} bases -> {n_data} oligos de datos")

    encoded = rs.encode_pool(oligos)
    extended = rs.add_column_parity(encoded, pool)
    n_parity = len(extended) - len(encoded)
    print(f"Oligos de paridad añadidos: {n_parity} "
          f"(total en el pool: {len(extended)})")

    sim = ChannelSimulator(
        sub_rate=sub_rate, ins_rate=ins_rate, del_rate=del_rate,
        dropout_rate=dropout_rate, seed=seed,
    )
    noisy, stats = sim.simulate(extended, reorder=True)
    print(f"Oligos perdidos en el canal: {stats.n_dropped} / {stats.n_oligos_in} "
          f"({stats.dropout_rate:.1%})")

    noisy.sort(key=lambda o: o.index)
    data_n = [o for o in noisy if o.index < n_data]
    parity_n = [o for o in noisy if o.index >= n_data]

    recovered_pool, n_rec = rs.recover_with_column_parity(
        data_n + parity_n, n_data, pool
    )
    print(f"Oligos recuperados vía paridad de columna: {n_rec}")

    decoded, report = rs.decode_pool(recovered_pool, orig_payload_len)
    print(report.summary())

    decoded.sort(key=lambda o: o.index)
    rebuilt, _ = pool.assemble(decoded)
    final = enc.decode_sequence(rebuilt[: len(master)], enc.start_bases)

    ok = final == image_bytes
    sha_ok = (
        hashlib.sha256(final).hexdigest()
        == hashlib.sha256(image_bytes).hexdigest()
    )
    print()
    print(f"Imagen recuperada == original: {ok}")
    print(f"SHA-256 verificado:            {sha_ok}")

    if ok and out_path:
        with open(out_path, "wb") as f:
            f.write(final)
        print(f"Imagen recuperada guardada en: {out_path}")

    return ok and sha_ok

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", help="Ruta a la imagen (o cualquier archivo binario)")
    parser.add_argument("--out", default=None, help="Ruta de salida para la imagen recuperada")
    parser.add_argument("--dropout", type=float, default=0.10, help="Tasa de dropout del canal (default 0.10)")
    parser.add_argument("--sub-rate", type=float, default=0.0, help="Tasa de sustitución de bases")
    parser.add_argument("--ins-rate", type=float, default=0.0, help="Tasa de inserción de bases")
    parser.add_argument("--del-rate", type=float, default=0.0, help="Tasa de deleción de bases")
    parser.add_argument("--seed", type=int, default=7, help="Semilla del simulador de canal")
    parser.add_argument("--oligo-len", type=int, default=150)
    parser.add_argument("--redundancy", type=float, default=0.30, help="Redundancia RS por-oligo")
    parser.add_argument("--col-redundancy", type=float, default=0.20, help="Redundancia RS por-columna")
    args = parser.parse_args()

    ok = encode_decode_image(
        args.image,
        out_path=args.out,
        dropout_rate=args.dropout,
        sub_rate=args.sub_rate,
        ins_rate=args.ins_rate,
        del_rate=args.del_rate,
        seed=args.seed,
        oligo_len=args.oligo_len,
        redundancy=args.redundancy,
        col_redundancy=args.col_redundancy,
    )
    return 0 if ok else 1

if __name__ == "__main__":
    sys.exit(main())
