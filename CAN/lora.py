"""
LoRa Backup Communication Simulation
Uses com0com virtual null-modem pair: COM10 (sender) <-> COM11 (receiver)
EmuNoise=0.01 is already active on the port — used for Bonus corruption simulation.

Install dependency:  pip install pyserial
"""

import serial
import threading
import time
import json
import zlib
import struct
import random

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
SENDER_PORT   = 'COM10'
RECEIVER_PORT = 'COM11'
BAUD_RATE     = 9600          # realistic LoRa UART baud rate

LORA_MAX_PAYLOAD   = 255      # max bytes per LoRa packet payload
LORA_BANDWIDTH_BPS = 100_000  # 100 kB/s

# ── Bonus: Software corruption (on top of com0com EmuNoise) ──────────────────
CORRUPTION_ENABLED = True   # Set True to enable extra software corruption
CORRUPTION_RATE    = 0.3     # 30% of packets get an extra bit-flip

# ─────────────────────────────────────────────
# PACKET HELPERS
# ─────────────────────────────────────────────
MAGIC = b'\xAB\xCD'

def build_packet(seq: int, payload: bytes) -> bytes:
    """
    Frame format:
      2 B  magic
      2 B  sequence number
      2 B  payload length
      N B  payload
      4 B  CRC32
    """
    header = MAGIC + struct.pack('>HH', seq, len(payload))
    crc    = struct.pack('>I', zlib.crc32(header + payload) & 0xFFFFFFFF)
    return header + payload + crc

def parse_packet(raw: bytes):
    """Returns (seq, payload) or raises ValueError on bad magic / CRC."""
    if len(raw) < 10:
        raise ValueError("Packet too short")
    if raw[:2] != MAGIC:
        raise ValueError("Bad magic bytes")
    seq, length = struct.unpack('>HH', raw[2:6])
    payload = raw[6:6 + length]
    received_crc = raw[6 + length: 10 + length]
    expected_crc = struct.pack('>I', zlib.crc32(raw[:6 + length]) & 0xFFFFFFFF)
    if received_crc != expected_crc:
        raise ValueError("CRC mismatch — data corrupted!")
    return seq, payload

def compress_data(data: dict) -> bytes:
    return zlib.compress(json.dumps(data).encode())

def decompress_data(raw: bytes) -> dict:
    return json.loads(zlib.decompress(raw).decode())

# ─────────────────────────────────────────────
# BONUS – Extra Software Corruption
# ─────────────────────────────────────────────

def maybe_corrupt(packet: bytes, seq: int) -> bytes:
    if CORRUPTION_ENABLED and random.random() < CORRUPTION_RATE:
        ba  = bytearray(packet)
        idx = random.randint(6, max(6, len(ba) - 5))
        ba[idx] ^= 0xFF
        return bytes(ba)
    return packet

# ─────────────────────────────────────────────
# SENDER  (writes to COM10)
# ─────────────────────────────────────────────
def sender(messages: list):
    print(f"\n[SENDER] Opening {SENDER_PORT} ...")
    with serial.Serial(SENDER_PORT, BAUD_RATE, timeout=2) as port:
        print(f"[SENDER] Connected on {port.name}")
        for seq, msg in enumerate(messages):
            compressed = compress_data(msg)

            # Fragment into LoRa-sized chunks
            chunks = [compressed[i:i + LORA_MAX_PAYLOAD]
                      for i in range(0, len(compressed), LORA_MAX_PAYLOAD)]

            for chunk in chunks:
                pkt = build_packet(seq, chunk)
                pkt = maybe_corrupt(pkt, seq)         # Bonus software corruption
                port.write(pkt)

                # Throttle to LoRa bandwidth
                time.sleep(len(pkt) / LORA_BANDWIDTH_BPS)

                print(f"[SENDER] Packet {seq + 1}/{len(messages)} sent")

        # END sentinel
        port.write(build_packet(0xFFFF, b'END'))
        print("[SENDER] All messages sent.")

# ─────────────────────────────────────────────
# RECEIVER  (reads from COM11)
# ─────────────────────────────────────────────
def receiver():
    print(f"\n[RECEIVER] Opening {RECEIVER_PORT} ...")
    with serial.Serial(RECEIVER_PORT, BAUD_RATE, timeout=5) as port:
        print(f"[RECEIVER] Listening on {port.name}")
        buf = b''
        received, errors = 0, 0

        while True:
            chunk = port.read(512)
            if not chunk:
                print("[RECEIVER] Timeout — no data received.")
                break
            buf += chunk

            # Parse all complete packets in the buffer
            while len(buf) >= 10:
                idx = buf.find(MAGIC)
                if idx == -1:
                    buf = b''
                    break
                buf = buf[idx:]          # discard bytes before magic

                if len(buf) < 10:
                    break
                _, length = struct.unpack('>HH', buf[2:6])
                pkt_len = 10 + length
                if len(buf) < pkt_len:
                    break                # wait for more bytes

                raw_pkt = buf[:pkt_len]
                buf     = buf[pkt_len:]

                try:
                    seq, payload = parse_packet(raw_pkt)
                    if payload == b'END':
                        print(f"\n[RECEIVER] END received.")
                        print(f"[STATS] Received OK={received}  Errors={errors}")
                        return
                    data = decompress_data(payload)
                    print(f"[RECEIVER] [{seq}] SUCCESS")
                    received += 1
                except (ValueError, zlib.error) as e:
                    errors += 1
                    print(f"  [RX ERROR] {e}  (total errors={errors})")

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == '__main__':
    # Sample telemetry data to transmit
    messages = [
        {"sensor": "temp",     "value": 23.5, "unit": "C",   "ts": 1700000001},
        {"sensor": "humidity", "value": 65.2, "unit": "%",   "ts": 1700000002},
        {"sensor": "pressure", "value": 1013, "unit": "hPa", "ts": 1700000003},
        {"sensor": "gps",      "lat": 51.5,   "lon": -0.1,   "ts": 1700000004},
        {"sensor": "battery",  "value": 87,   "unit": "%",   "ts": 1700000005},
        {"sensor": "temp",      "value": 24.1, "unit": "C",   "ts": 1700000006},
        {"sensor": "humidity",  "value": 63.8, "unit": "%",   "ts": 1700000007},
        {"sensor": "pressure",  "value": 1015, "unit": "hPa", "ts": 1700000008},
        {"sensor": "gps",       "lat": 51.501, "lon": -0.101, "ts": 1700000009},
        {"sensor": "battery",   "value": 86,   "unit": "%",   "ts": 1700000010},
        {"sensor": "temp",      "value": 22.9, "unit": "C",   "ts": 1700000011},
        {"sensor": "humidity",  "value": 67.1, "unit": "%",   "ts": 1700000012},
        {"sensor": "pressure",  "value": 1012, "unit": "hPa", "ts": 1700000013},
        {"sensor": "gps",       "lat": 51.502, "lon": -0.102, "ts": 1700000014},
        {"sensor": "battery",   "value": 85,   "unit": "%",   "ts": 1700000015},
        {"sensor": "temp",      "value": 25.3, "unit": "C",   "ts": 1700000016},
        {"sensor": "humidity",  "value": 60.5, "unit": "%",   "ts": 1700000017},
        {"sensor": "pressure",  "value": 1010, "unit": "hPa", "ts": 1700000018},
        {"sensor": "gps",       "lat": 51.503, "lon": -0.103, "ts": 1700000019},
        {"sensor": "battery",   "value": 84,   "unit": "%",   "ts": 1700000020},
        {"sensor": "temp",      "value": 21.7, "unit": "C",   "ts": 1700000021},
        {"sensor": "humidity",  "value": 70.3, "unit": "%",   "ts": 1700000022},
        {"sensor": "pressure",  "value": 1008, "unit": "hPa", "ts": 1700000023},
        {"sensor": "gps",       "lat": 51.504, "lon": -0.104, "ts": 1700000024},
        {"sensor": "battery",   "value": 83,   "unit": "%",   "ts": 1700000025},
        {"sensor": "temp",      "value": 26.0, "unit": "C",   "ts": 1700000026},
        {"sensor": "humidity",  "value": 58.9, "unit": "%",   "ts": 1700000027},
        {"sensor": "pressure",  "value": 1016, "unit": "hPa", "ts": 1700000028},
        {"sensor": "gps",       "lat": 51.505, "lon": -0.105, "ts": 1700000029},
        {"sensor": "battery",   "value": 82,   "unit": "%",   "ts": 1700000030},
        {"sensor": "temp",      "value": 23.2, "unit": "C",   "ts": 1700000031},
        {"sensor": "humidity",  "value": 64.4, "unit": "%",   "ts": 1700000032},
        {"sensor": "pressure",  "value": 1011, "unit": "hPa", "ts": 1700000033},
        {"sensor": "gps",       "lat": 51.506, "lon": -0.106, "ts": 1700000034},
        {"sensor": "battery",   "value": 81,   "unit": "%",   "ts": 1700000035},
        {"sensor": "temp",      "value": 27.4, "unit": "C",   "ts": 1700000036},
        {"sensor": "humidity",  "value": 55.0, "unit": "%",   "ts": 1700000037},
        {"sensor": "pressure",  "value": 1009, "unit": "hPa", "ts": 1700000038},
        {"sensor": "gps",       "lat": 51.507, "lon": -0.107, "ts": 1700000039},
        {"sensor": "battery",   "value": 80,   "unit": "%",   "ts": 1700000040},
        {"sensor": "temp",      "value": 20.5, "unit": "C",   "ts": 1700000041},
        {"sensor": "humidity",  "value": 72.6, "unit": "%",   "ts": 1700000042},
        {"sensor": "pressure",  "value": 1014, "unit": "hPa", "ts": 1700000043},
        {"sensor": "gps",       "lat": 51.508, "lon": -0.108, "ts": 1700000044},
        {"sensor": "battery",   "value": 79,   "unit": "%",   "ts": 1700000045},
        {"sensor": "temp",      "value": 24.8, "unit": "C",   "ts": 1700000046},
        {"sensor": "humidity",  "value": 61.7, "unit": "%",   "ts": 1700000047},
        {"sensor": "pressure",  "value": 1017, "unit": "hPa", "ts": 1700000048},
        {"sensor": "gps",       "lat": 51.509, "lon": -0.109, "ts": 1700000049},
        {"sensor": "battery",   "value": 78,   "unit": "%",   "ts": 1700000050},
    ]

    print("=" * 55)
    print(" LoRa Communication Simulation via com0com")
    print(f" Sender   -> {SENDER_PORT}")
    print(f" Receiver <- {RECEIVER_PORT}")
    print(f" EmuNoise=0.01 active on port (hardware noise)")
    print(f" Software corruption: {'ON' if CORRUPTION_ENABLED else 'OFF'}")
    print("=" * 55)

    # Start receiver in a background thread
    rx_thread = threading.Thread(target=receiver, daemon=True)
    rx_thread.start()

    time.sleep(0.5)    # give receiver time to open the port
    sender(messages)

    rx_thread.join(timeout=10)

