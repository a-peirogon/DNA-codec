"""
encoder.py — Binary file → ACGT DNA sequence encoder.

Encoding pipeline
-----------------
  1. Read the input file as raw bytes.
  2. Prepend an 8-byte header:
       [4 bytes: magic 0xDEAD_BEEF] [4 bytes: original file length in bytes]
  3. Compute SHA-256 digest of the raw bytes; append 32 bytes at the end.
     This lets the decoder verify integrity after reconstruction.
  4. Pad the byte stream to an even number of bytes (required for 2-bit mapping).
  5. Convert every byte to four 2-bit symbols:
       byte b → [(b>>6)&3, (b>>4)&3, (b>>2)&3, b&3]
  6. Apply rotation encoding (see constraints.ConstraintEnforcer) to map
     2-bit symbols → ACGT bases while naturally avoiding homopolymer runs.
  7. The optimal start_base (one of A/C/G/T) is chosen per-block of 200 nt
     to keep GC content near 50 %.  The start_base sequence is stored in the
     header of each oligo (handled by oligos.py).

References
----------
  Church, G.M. et al. (2012). Next-generation digital information storage
    in DNA. Science, 337(6102), 1628–1628.
  Goldman, N. et al. (2013). Towards practical, high-capacity, low-maintenance
    information storage in synthesised DNA. Nature, 494(7435), 77–80.
  Organick, L. et al. (2018). Random access in large-scale DNA data storage.
    Nature Biotechnology, 36(3), 242–248.

Public API
----------
  DNAEncoder(block_size=200)          — construct encoder
  encoder.encode_file(path) → str     — returns master ACGT sequence
  encoder.encode_bytes(data) → str    — encode raw bytes
  DNAEncoder.decode_bytes(seq) → bytes — static, reverses encode_bytes
"""

from __future__ import annotations

import hashlib
import os
import struct
from pathlib import Path
from typing import Union

from .constraints import ConstraintChecker, ConstraintEnforcer, _BASE_TO_BITS

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAGIC: bytes = b"\xDE\xAD\xBE\xEF"
HEADER_FMT: str = ">4sI"          # big-endian: 4-byte magic + 4-byte uint32
HEADER_SIZE: int = struct.calcsize(HEADER_FMT)   # 8 bytes
DIGEST_SIZE: int = 32             # SHA-256
BITS_PER_BASE: int = 2
BITS_PER_BYTE: int = 8
BASES_PER_BYTE: int = BITS_PER_BYTE // BITS_PER_BASE  # 4


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def bytes_to_dibits(data: bytes) -> list[int]:
    """
    Flatten bytes into a list of 2-bit integers (dibits).

    Each byte b produces four dibits in MSB-first order:
        [ (b >> 6) & 3,  (b >> 4) & 3,  (b >> 2) & 3,  b & 3 ]
    """
    dibits: list[int] = []
    for byte in data:
        dibits.append((byte >> 6) & 3)
        dibits.append((byte >> 4) & 3)
        dibits.append((byte >> 2) & 3)
        dibits.append(byte & 3)
    return dibits


def dibits_to_bytes(dibits: list[int]) -> bytes:
    """
    Reconstruct bytes from a flat list of 2-bit integers.

    Raises ValueError if len(dibits) is not a multiple of 4.
    """
    if len(dibits) % 4 != 0:
        raise ValueError(
            f"dibits length {len(dibits)} is not a multiple of 4"
        )
    result: list[int] = []
    for i in range(0, len(dibits), 4):
        byte = (
            (dibits[i]     << 6)
            | (dibits[i+1] << 4)
            | (dibits[i+2] << 2)
            |  dibits[i+3]
        )
        result.append(byte)
    return bytes(result)


# ---------------------------------------------------------------------------
# Encoder class
# ---------------------------------------------------------------------------

class DNAEncoder:
    """
    Encodes arbitrary binary data to a master ACGT sequence.

    Parameters
    ----------
    block_size : int
        Number of DNA bases in each rotation-encoding block.  The optimal
        start_base is chosen independently per block to maximise GC balance.
        Smaller blocks → better GC control; larger blocks → fewer metadata bits.
        Default: 200 nt (covers one full oligo payload).
    gc_min, gc_max : float
        Target GC content bounds forwarded to the constraint checker.
    max_homopolymer : int
        Maximum consecutive identical bases allowed.
    """

    def __init__(
        self,
        block_size: int = 200,
        gc_min: float = 0.40,
        gc_max: float = 0.60,
        max_homopolymer: int = 3,
        max_palindrome: int = 6,
    ) -> None:
        self.block_size = block_size
        self.checker = ConstraintChecker(
            gc_min=gc_min,
            gc_max=gc_max,
            max_homopolymer=max_homopolymer,
            max_palindrome=max_palindrome,
        )
        self.enforcer = ConstraintEnforcer(self.checker)
        # Populated after encode_bytes is called
        self.start_bases: list[str] = []

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def encode_file(self, path: Union[str, Path]) -> str:
        """
        Read *path* and return the master ACGT sequence.

        Also stores self.start_bases (one entry per block) and
        self.sha256 (hex digest of original raw bytes).
        """
        data = Path(path).read_bytes()
        return self.encode_bytes(data)

    def encode_bytes(self, data: bytes) -> str:
        """
        Encode raw *data* bytes to a master ACGT sequence.

        Steps:
          1. Compute SHA-256 of raw data.
          2. Build payload: header + data + digest.
          3. Pad to align to full bytes (already byte-aligned; padding for
             even BASES_PER_BYTE alignment is inherent).
          4. Convert payload to dibits.
          5. Encode each block of (block_size) dibits with best start_base.

        Returns
        -------
        str : Master ACGT sequence.
        """
        # 1. Digest
        digest = hashlib.sha256(data).digest()
        self.sha256 = hashlib.sha256(data).hexdigest()

        # 2. Build payload bytes: header | data | sha256_digest
        header = struct.pack(HEADER_FMT, MAGIC, len(data))
        payload: bytes = header + data + digest

        # 3. Pad payload so that len(payload) * BASES_PER_BYTE is divisible
        #    by block_size (simplifies block chunking).
        total_bases_needed = len(payload) * BASES_PER_BYTE
        remainder = total_bases_needed % self.block_size
        if remainder != 0:
            pad_bases = self.block_size - remainder
            # pad_bases must be divisible by BASES_PER_BYTE
            pad_bytes_needed = (pad_bases + BASES_PER_BYTE - 1) // BASES_PER_BYTE
            payload += b"\x00" * pad_bytes_needed

        # 4. Bytes → dibits
        dibits = bytes_to_dibits(payload)

        # 5. Encode in blocks
        self.start_bases = []
        dna_parts: list[str] = []

        for block_start in range(0, len(dibits), self.block_size):
            block = dibits[block_start : block_start + self.block_size]
            start_base = self.enforcer.best_start_base(block)
            self.start_bases.append(start_base)
            dna_parts.append(
                self.enforcer.rotation_encode(block, start_base=start_base)
            )

        master_seq = "".join(dna_parts)
        return master_seq

    # ------------------------------------------------------------------
    # Decoding (reverse pipeline) — static so decoder can call it
    # ------------------------------------------------------------------

    def decode_sequence(self, seq: str, start_bases: list[str]) -> bytes:
        """
        Reverse encode_bytes: ACGT sequence → original raw bytes.

        Parameters
        ----------
        seq : str
            Master ACGT sequence (output of encode_bytes).
        start_bases : list[str]
            One start_base per block of self.block_size bases.

        Returns
        -------
        bytes : Original file content (without header or digest).

        Raises
        ------
        ValueError : If the SHA-256 digest embedded in the payload does not
                     match the decoded data.
        """
        enforcer = self.enforcer
        block_size = self.block_size

        # 1. Recover dibits block by block
        dibits: list[int] = []
        for i, start_base in enumerate(start_bases):
            block_seq = seq[i * block_size : (i + 1) * block_size]
            dibits.extend(enforcer.rotation_decode(block_seq, start_base=start_base))

        # 2. dibits → bytes
        # Trim dibits to multiple of 4 (safety)
        trim = len(dibits) - (len(dibits) % 4)
        payload = dibits_to_bytes(dibits[:trim])

        # 3. Parse header
        magic, data_len = struct.unpack(HEADER_FMT, payload[:HEADER_SIZE])
        if magic != MAGIC:
            raise ValueError(f"Bad magic bytes: {magic!r} (expected {MAGIC!r})")

        # 4. Extract data and digest
        data = payload[HEADER_SIZE : HEADER_SIZE + data_len]
        stored_digest = payload[HEADER_SIZE + data_len : HEADER_SIZE + data_len + DIGEST_SIZE]

        # 5. Verify integrity
        actual_digest = hashlib.sha256(data).digest()
        if actual_digest != stored_digest:
            raise ValueError(
                "SHA-256 mismatch — data corrupted during encode/decode cycle.\n"
                f"  stored:  {stored_digest.hex()}\n"
                f"  actual:  {actual_digest.hex()}"
            )

        return data

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def constraint_report(self, seq: str, sample_size: int = 5) -> dict:
        """
        Run constraint checks on *sample_size* evenly spaced windows of
        `block_size` bases and return a summary dict.
        """
        n = len(seq)
        step = max(1, n // sample_size)
        reports = []
        for start in range(0, n - self.block_size + 1, step):
            window = seq[start : start + self.block_size]
            reports.append(self.checker.check(window))

        gc_values = [r.gc_content for r in reports]
        return {
            "total_bases": n,
            "blocks_checked": len(reports),
            "gc_mean": sum(gc_values) / len(gc_values) if gc_values else 0.0,
            "gc_min": min(gc_values) if gc_values else 0.0,
            "gc_max": max(gc_values) if gc_values else 0.0,
            "all_passed": all(r.passed for r in reports),
            "violations": [str(r) for r in reports if not r.passed],
        }
