import serial
import numpy as np
import threading
import time

# Simulation of virtual serial ports

# Mac: First install socat using, brew install socat
# Run the following command in the terminal and keep it running on the execution of code
# socat -d -d pty,raw,echo=0 pty,raw,echo=0
# You can find the port names in the terminal where you ran the socat command. 

# Windows: In this application, we are going to use com0com
# There will a zip file attched to the resouces of the application. Run setup.exe
# Open your installed folder, run setupc.exe
# Type: install PortName=COM30 PortName=COM31 to create your virtual ports

# Replace "your_port" with the port name of the sender and receiver

# To install dependencies, 
# pip install pyserial
# pip install numpy

# Now to run your code, 
# python3 application.py

port_sender = "COM10"  
port_receiver = "COM11" 

BYTE_RESET_PROBABLITY = 0.005

# ── Protocol constants ────────────────────────────────────────────────────────
START_BYTE = 0xAA
END_BYTE   = 0xFF

def send_data(ser: serial.Serial, data: np.ndarray) -> None:

    data_to_send = []

    # Frame format: [START, len_high, len_low, d0, d1, ..., dn, checksum, END]
    length   = len(data)
    checksum = int(np.sum(data) % 256)

    data_to_send.append(START_BYTE)
    data_to_send.append((length >> 8) & 0xFF)   # length high byte
    data_to_send.append(length & 0xFF)           # length low byte
    for val in data:
        data_to_send.append(int(val))
    data_to_send.append(checksum)
    data_to_send.append(END_BYTE)

    # For challenge question, uncomment this
    # This should be done, JUST before sending the data
    # No other code should be there after this, other than sending the data itself

    for i in range(len(data_to_send)):
        if np.random.random() < BYTE_RESET_PROBABLITY:
            data_to_send[i] = 0x00

    for byte in data_to_send:
        ser.write(bytes([byte]))

def receive_data(ser: serial.Serial) -> tuple[np.ndarray, bool]:

    received_pwm_data = []
    acknowledgement = False

    # Wait for START byte
    while True:
        b = ser.read(1)
        if not b:
            return np.array(received_pwm_data), False   # timeout
        if b[0] == START_BYTE:
            break

    # Read length (2 bytes)
    len_bytes = ser.read(1) + ser.read(1)
    if len(len_bytes) < 2:
        return np.array(received_pwm_data), False
    length = (len_bytes[0] << 8) | len_bytes[1]

    # Read payload
    for _ in range(length):
        b = ser.read(1)
        if not b:
            return np.array(received_pwm_data), False
        received_pwm_data.append(b[0])

    # Read checksum
    chk_byte = ser.read(1)
    if not chk_byte:
        return np.array(received_pwm_data), False
    received_checksum = chk_byte[0]

    # Read END byte
    end_byte = ser.read(1)
    if not end_byte or end_byte[0] != END_BYTE:
        return np.array(received_pwm_data), False

    # Verify checksum
    expected_checksum = int(np.sum(received_pwm_data) % 256)
    if received_checksum == expected_checksum and len(received_pwm_data) == length:
        acknowledgement = True

    return np.array(received_pwm_data), acknowledgement

def receive_thread_task(received_data: list, no_of_success: int):
    no_of_tries = 0
    try:
        with serial.Serial(port_receiver, 9600, timeout=0.2) as ser:
            while len(received_data) < 100 and no_of_tries < 150:
                pass

                received_arr, acknowledgement = receive_data(ser)
                
                if np.any(received_arr) or acknowledgement:
                    received_data.append(received_arr)
                    if acknowledgement:
                        print(f"[RECEIVER] [{len(received_data)}] SUCCESS")
                        no_of_success[0] += 1
                    else: 
                        print(f"[RECEIVER] [{len(received_data)}] FAILED")
                no_of_tries += 1
                time.sleep(0.01) 
            else: 
                print(f"[RECEIVER] Time Out")
    except Exception as e:
        print(f"Receiver Thread Error: {e}")

def send_thread_task(all_data):
    try: 
        with serial.Serial(port_sender, 9600) as ser:
            for i, data in enumerate(all_data):
                send_data(ser, data)
                print(f"[SENDER]   [{i+1}] Packet Sent")
                time.sleep(0.15) 
    except Exception as e:
        print(f"Sender Thread Error: {e}")

def generate_pwm():

    pwm = np.random.randint(0, 255, size=(100,))
    return pwm

def main():

    pwm_data = [generate_pwm() for i in range(100)]
    received_data = []
    no_of_success = [0]

    receiver_thread = threading.Thread(target=receive_thread_task, args=(received_data, no_of_success))
    receiver_thread.daemon = True
    receiver_thread.start()

    time.sleep(1)

    sender_thread = threading.Thread(target=send_thread_task, args=(pwm_data,))
    sender_thread.daemon = True
    sender_thread.start()

    time.sleep(1)

    sender_thread.join(timeout=30)
    receiver_thread.join(timeout=30)

    print(f"Total Successful: {no_of_success[0]}/100")

if __name__ == "__main__":
    main()
