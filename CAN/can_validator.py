"""
CAN Frame Validator
====================
Validates 11-bit (base frame) CAN frames from a CSV file.

Checks performed:
  1. ID validity                  - 11-bit IDs must be in range 0x000-0x7FF
  2. DLC validity                 - DLC must be 0-8
  3. DLC / data length mismatch   - actual number of data bytes must equal DLC
  4. CRC-15 correctness           - computed over the CAN base-frame bit stream
                                    using polynomial 0x4599

CAN base-frame bit stream used for CRC-15:
  SOF (1 bit, 0) | ID (11 bits) | RTR (1 bit) | IDE (1 bit) | r0 (1 bit) | DLC (4 bits) | DATA (DLC x 8 bits)

CRC-15 algorithm:
  - Processes bits MSB-first
  - Feedback based on the MSB of the *current* register XOR the incoming bit
    (standard LFSR: if top-bit XOR in-bit == 1, shift and XOR poly; else just shift)
  - Initial value: 0x0000
  - No final XOR or reflection

Usage:
  python can_validator.py <csv_file>
  python can_validator.py          <- uses can_frames.csv in same directory
"""

import csv
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# CRC-15 / CAN   (polynomial 0x4599)
# ---------------------------------------------------------------------------
CRC_POLY = 0x4599   # x^15 + x^14 + x^10 + x^8 + x^7 + x^4 + x^3 + 1
CRC_BITS = 15
CRC_MASK = (1 << CRC_BITS) - 1   # 0x7FFF


def crc15(bit_stream: list) -> int:
    """
    CRC-15/CAN over a list of ints (each 0 or 1), MSB-first.
    Feedback uses the MSB of the *current* register before shifting.
    """
    crc = 0
    for bit in bit_stream:
        if ((crc >> (CRC_BITS - 1)) ^ (bit & 1)) & 1:
            crc = ((crc << 1) ^ CRC_POLY) & CRC_MASK
        else:
            crc = (crc << 1) & CRC_MASK
    return crc


def int_to_bits(value: int, width: int) -> list:
    """Integer -> list of bits, MSB first, fixed width."""
    return [(value >> (width - 1 - i)) & 1 for i in range(width)]


# ---------------------------------------------------------------------------
# Build CAN base-frame bit stream for CRC computation
# Bit order: SOF | ID[10:0] | RTR | IDE | r0 | DLC[3:0] | DATA bytes
# ---------------------------------------------------------------------------
def build_frame_bits(can_id: int, ide: int, rtr: int,
                     dlc: int, data_bytes: list) -> list:
    bits  = [0]                          # SOF (dominant)
    bits += int_to_bits(can_id, 11)      # 11-bit identifier
    bits += [rtr & 1]                    # RTR
    bits += [ide & 1]                    # IDE  (0 = base frame)
    bits += [0]                          # r0   (reserved, dominant)
    bits += int_to_bits(dlc, 4)          # DLC
    for byte in data_bytes:              # data field
        bits += int_to_bits(byte, 8)
    return bits


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
MAX_ID  = 0x7FF   # 11-bit limit
MAX_DLC = 8


def validate_frame(row: dict) -> dict:
    """
    Validate one CSV row.  Returns a result dict.

    The 'crc' column contains the CRC that was transmitted with the frame
    (which may be correct or corrupted).  The validator recomputes the expected
    CRC and compares; any difference is flagged as bad_crc.
    """
    ts       = row["timestamp"]
    id_str   = row["id"].strip()
    ide      = int(row["ide"].strip())
    rtr      = int(row["rtr"].strip())
    dlc      = int(row["dlc"].strip())
    data_str = row["data"].strip()
    crc_str  = row["crc"].strip()
    csv_errs = row["errors"].strip()

    # Parse CAN ID
    try:
        can_id = int(id_str, 16)
    except ValueError:
        can_id = -1

    # Parse data bytes
    if data_str:
        try:
            data_bytes = [int(b, 16) for b in data_str.split()]
        except ValueError:
            data_bytes = []
    else:
        data_bytes = []

    # Parse transmitted CRC
    try:
        crc_transmitted = int(crc_str, 16)
    except ValueError:
        crc_transmitted = -1

    detected = []

    # Check 1: ID range (must fit in 11 bits)
    if can_id < 0 or can_id > MAX_ID:
        detected.append("bad_id")

    # Check 2: DLC range (0-8)
    if not (0 <= dlc <= MAX_DLC):
        detected.append("bad_dlc")

    # Check 3: DLC vs actual data byte count
    # Only meaningful when DLC itself is valid
    if (0 <= dlc <= MAX_DLC) and (len(data_bytes) != dlc):
        detected.append("mismatch_of_dlc_and_data_frame")

    # Check 4: CRC-15
    # Compute over the actual (possibly malformed) data present in the CSV.
    # Use can_id=0 if the ID was invalid so the CRC computation still runs.
    effective_id = can_id if can_id >= 0 else 0
    frame_bits   = build_frame_bits(effective_id, ide, rtr, dlc, data_bytes)
    crc_computed = crc15(frame_bits)

    if crc_computed != crc_transmitted:
        detected.append("bad_crc")

    status = "PASS" if not detected else "FAIL"

    return {
        "timestamp":           ts,
        "id":                  id_str,
        "ide":                 ide,
        "rtr":                 rtr,
        "dlc":                 dlc,
        "data":                data_str,
        "crc_transmitted":     f"{crc_transmitted:04X}",
        "crc_computed":        f"{crc_computed:04X}",
        "detected_errors":     detected if detected else ["none"],
        "csv_reported_errors": csv_errs,
        "status":              status,
    }


# ---------------------------------------------------------------------------
# Report printer
# ---------------------------------------------------------------------------
SEP = "=" * 80


def print_report(results: list):
    total  = len(results)
    passes = sum(1 for r in results if r["status"] == "PASS")
    fails  = total - passes

    agreement = sum(
        1 for r in results
        if ",".join(sorted(r["detected_errors"])) ==
           ",".join(sorted(r["csv_reported_errors"].split(",")))
    )

    print(SEP)
    print("  CAN FRAME VALIDATION REPORT")
    print(SEP)
    print(f"  Total frames                   : {total}")
    print(f"  PASS (no errors detected)      : {passes}")
    print(f"  FAIL (one or more errors)      : {fails}")
    print(f"  Agreement with CSV error labels: {agreement}/{total}")
    print(SEP)

    for i, r in enumerate(results, 1):
        det_str = ", ".join(r["detected_errors"])
        rep_str = r["csv_reported_errors"]
        agree   = "OK" if det_str == rep_str else "DIFFERS"
        crc_ok  = r["crc_transmitted"] == r["crc_computed"]

        print(f"\nFrame #{i:>2}  [{r['status']}]")
        print(f"  Timestamp     : {r['timestamp']}")
        print(f"  ID            : {r['id']}   IDE={r['ide']}  RTR={r['rtr']}  DLC={r['dlc']}")
        print(f"  Data          : {r['data'] if r['data'] else '(none)'}")
        print(
            f"  CRC (file)    : {r['crc_transmitted']}  "
            f"CRC (computed)  : {r['crc_computed']}  "
            f"{'[match]' if crc_ok else '[MISMATCH]'}"
        )
        print(f"  Detected      : {det_str}")
        print(f"  CSV errors    : {rep_str}  [{agree}]")

    print(f"\n{SEP}")
    print("  Notes:")
    print("   - CRC is computed over: SOF + ID(11b) + RTR + IDE + r0 + DLC(4b) + DATA")
    print("   - Polynomial 0x4599, init=0, MSB-first, no final XOR")
    print("   - 'bad_id'  : ID > 0x7FF (exceeds 11-bit range)")
    print("   - 'bad_dlc' : DLC > 8 or < 0")
    print("   - 'mismatch_of_dlc_and_data_frame' : len(data) != DLC")
    print("   - 'bad_crc' : computed CRC != CRC in file")
    print(SEP)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main(csv_path: str):
    path = Path(csv_path)
    if not path.exists():
        print(f"[ERROR] File not found: {csv_path}")
        sys.exit(1)

    results = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append(validate_frame(row))

    print_report(results)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        default  = "can_frames.csv"
        fallback = "/mnt/user-data/uploads/can_frames.csv"
        target   = default if Path(default).exists() else fallback
        print(f"Usage: python can_validator.py <csv_file>")
        print(f"No argument provided -- running with: {target}\n")
        main(target)
    else:
        main(sys.argv[1])
