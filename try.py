import pyaudio
import threading
import time
import numpy as np
import math
import random
from queue import Queue
from scipy.fft import fft

F_0 = 600
F_1 = 900
F_NULL = 1200
BIT_RATE = 15
FS = 44100
DURATION_PER_BIT = 1 / BIT_RATE
FORMAT = pyaudio.paFloat32
CHANNELS = 1
CHUNK = int(FS * DURATION_PER_BIT)
epsilon = 30 #for extracting frequencies that got deviated due to noise

def detect_frequency(signal):
    signal_fft = fft(signal)
    freqs = np.fft.fftfreq(len(signal_fft), 1 / FS)
    
    peak_freq = abs(freqs[np.argmax(np.abs(signal_fft))])
    return peak_freq

def frequency_to_bit(frequency):
    if abs(frequency - F_0) < epsilon:
        return '0'
    elif abs(frequency - F_1) < epsilon:
        return '1'
    elif abs(frequency - F_NULL) < epsilon:
        return '#'
    else:
        return None

def receive_signal(duration): # This function is extracted from ChatGpt
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=FS, input=True, frames_per_buffer=CHUNK)

    print("Receiving signal...")
    start_time = time.time()
    received_bits = []

    while (time.time() - start_time < duration):
        data = stream.read(CHUNK)

        signal = np.frombuffer(data, dtype=np.float32)
        
        detected_freq = detect_frequency(signal)
        
        bit = frequency_to_bit(detected_freq)
        if bit != None:
            received_bits.append(bit)
        print(f"Received bit: {bit}, {detected_freq}")

    stream.stop_stream()
    stream.close()
    p.terminate()

    received_bits = ''.join(received_bits)

    i = 0
    start = 0
    end = 0

    while i < len(received_bits): # used to extract the bit string excluding #'s from the transmitted message.
        if received_bits[i] != '#':
            i = i + 1
            continue
        while(received_bits[i] == '#'):
            i = i+1
        start = i
        while(i < len(received_bits) and received_bits[i] != '#'):
            i = i + 1
        end = i
        break

    result = received_bits[start:end]

    return result

# Shared resource to simulate the busy state of the channel

def is_channel_busy():
    received_signal = receive_signal(3)
    return bool(received_signal)

def send_audio(message):
    bit_sequence = "####" + message + "####"

    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=FS, output=True, frames_per_buffer=CHUNK)

    t = np.linspace(0, DURATION_PER_BIT, int(FS * DURATION_PER_BIT), endpoint=False)


    for bit in bit_sequence:
        if bit == '0':
        
            tone = 0.5 * np.sin(2 * np.pi * F_0 * t)
        elif bit == '#':

            tone = 0.5 * np.sin(2 * np.pi * F_NULL * t)
        else:
        
            tone = 0.5 *np.sin(2 * np.pi * F_1 * t)
        
        tone = tone.astype(np.float32)
        stream.write(tone.tobytes())

    stream.stop_stream()
    stream.close()
    p.terminate()

# # Function to simulate clock synchronization (e.g., via NTP)
# def synchronize_clocks():
#     print("Clock synchronized.")

class Node:
    def _init_(self, node_id, messages):
        self.node_id = node_id
        self.message_queue = Queue()
        for msg in messages:
            self.message_queue.put(msg)
        self.received_messages = []

    def create_mac_header(self, message_type, sender_id, receiver_id, data=None):
        header = f"{message_type}{format(sender_id, '02b')}{format(receiver_id, '02b')}"
        if data:
            return header + data
        return header

    def send_rts(self, destination_id):
        if is_channel_busy():
            print(f"Node {self.node_id}: Channel is busy, cannot send RTS.")
            return False
        
        rts_message = self.create_mac_header('00', self.node_id, destination_id)
        print(f"Node {self.node_id}: Sending RTS to Node {destination_id}. RTS message: {rts_message}")
        send_audio(rts_message)

        return True

    def receive_rts(self, source_id):
        print(f"Node {self.node_id}: Received RTS from Node {source_id}. Checking for collision...")
        final = receive_signal(3)
        if len(final) == 6:
            first_two_bits = final[:2]
            last_two_bits = final[-2:]

            if first_two_bits == '00':
                last_two_decimal = int(last_two_bits, 2)

            if last_two_decimal == source_id:
                print(f"Node {self.node_id}: No collision detected. Sending CTS.")
                self.send_cts(source_id)
                return True
            else:
                time.sleep(3)
                return True
        else:
            print(f"Node {self.node_id}: Collision detected. Not sending CTS.")
            return False

    def send_cts(self, source_id):
        cts_message = self.create_mac_header('01', self.node_id, source_id)
        print(f"Node {self.node_id}: Sending CTS to Node {source_id}. CTS message: {cts_message}")
        send_audio(cts_message)

        return True
        
    def receive_cts(self, source_id,message):
        print(f"Node {self.node_id}: Listening for CTS from Node {source_id}...")
        final = receive_signal(3)
        if len(final) == 6:
            first_two_bits = final[:2]
            last_two_bits = final[-2:]
            middle_two_bits = final[2:4]
            destination_id = int(middle_two_bits,2)

            if first_two_bits == '01':
                last_two_decimal = int(last_two_bits, 2)

            if last_two_decimal == source_id:
                print(f"Node {self.node_id}: Received CTS.")
                self.send_data(destination_id,message)
                return True
            else:
                time.sleep(3)
                return True
        else:
            print(f"Node {self.node_id}: Collision detected. Not received CTS.")
            return False

    def send_data(self, destination_id, message):
        data_message = self.create_mac_header('10', self.node_id, destination_id, message)
        print(f"Node {self.node_id}: Sending data '{message}' to Node {destination_id}. Data message: {data_message}")
        send_audio(data_message)

        return True

    def receive_data(self):
        data_received = receive_signal(3)
        first_two_bits = data_received[:2]
        middle_two_bits = data_received[2:4]
        source_id = int(middle_two_bits,2)
        message = data_received[6:]
        print(f"Node {self.node_id}: Received data '{message}' from Node {source_id}. Sending ACK.")

        if first_two_bits == '10':
            print(f"Node {self.node_id}: Data received from Node {source_id}.")
            self.send_ack(self,source_id)
            return True
        else:
            print(f"Node {self.node_id}: No Data received. Collision likely occurred.")
            return False

    def send_ack(self, source_id):
        ack_message = self.create_mac_header('11', self.node_id, source_id)
        print(f"Node {self.node_id}: Sending ACK to Node {source_id}. ACK message: {ack_message}")
        send_audio(ack_message)

        return True
        
    def receive_ack(self, source):
        print(f"Node {self.node_id}: Waiting for ACK from Node {source.node_id}...")
        ack_received = random.choice([True, False])
        if ack_received:
            print(f"Node {self.node_id}: ACK received from Node {source.node_id}.")
            return True
        else:
            print(f"Node {self.node_id}: No ACK received. Collision likely occurred.")
            return False

# Sender thread: Handle sending messages using CSMA/CA
def sender_thread(node, receiver, message_trigger_queue):
    while True:
        if not message_trigger_queue.empty():
            message = message_trigger_queue.get()

            # Continuously sense the channel
            while True:
                if channel_busy.locked():
                    print(f"Node {node.node_id}: Channel is busy, waiting...")
                    time.sleep(0.5)  # Wait before sensing again
                else:
                    break  # Channel is free, proceed to send

            # Send RTS when the channel is free
            if node.send_rts(receiver):
                if not node.receive_cts():
                    # No CTS
                    backoff_time = random.uniform(1, 3)
                    print(f"Node {node.node_id}: Backing off for {backoff_time:.2f} seconds.")
                    time.sleep(backoff_time)
                    continue  # Retry sending

                # Proceed to send data
                node.send_data(receiver, message)

                if not node.receive_ack():
                    # No ACK received, collision likely occurred, implement backoff
                    backoff_time = random.uniform(1, 3)
                    print(f"Node {node.node_id}: No ACK received, backing off for {backoff_time:.2f} seconds.")
                    time.sleep(backoff_time)
                    continue  # Retry the message after backoff

                else:
                    break

    time.sleep(1)  # Simulate some delay between transmissions

# Receiver thread: Handle receiving messages
def receiver_thread(node):
    while True:
        # Listen for an RTS (simulated)
        print(f"Node {node.node_id}: Listening for RTS...")
        time.sleep(random.uniform(1, 2))  # Simulate random listening intervals
        
        if not channel_busy.locked():
            sender_id = random.randint(0, 3)  # Simulate receiving an RTS from another node
            if sender_id != node.node_id:
                print(f"Node {node.node_id}: RTS received from Node {sender_id}.")
                
                # Check if the channel is busy or collision detected
                if random.choice([True, False]):  # Simulate collision detection
                    print(f"Node {node.node_id}: Collision detected, not sending CTS.")
                else:
                    # Send CTS if no collision detected
                    print(f"Node {node.node_id}: No collision detected. Sending CTS to Node {sender_id}.")
                    cts_message = node.create_mac_header('01', node.node_id, sender_id)
                    print(f"Node {node.node_id}: CTS message: {cts_message}")

                    # Wait for data from the sender
                    time.sleep(1.5)  # Simulate wait for data transmission
                    print(f"Node {node.node_id}: Waiting for data from Node {sender_id}...")

                    # Simulate receiving data
                    if node.receive_data(sender, message):
                        data_message = "Simulated Data"  # Assume data is received correctly
                        print(f"Node {node.node_id}: Data received from Node {sender_id}: '{data_message}'.")
                    
                    else:
                        continue
                    
                    # Send ACK to the sender
                    print(f"Node {node.node_id}: Sending ACK to Node {sender_id}.")
                    ack_message = node.create_mac_header('11', node.node_id, sender_id)
                    print(f"Node {node.node_id}: ACK message: {ack_message}")

        time.sleep(1)  # Delay between listening attempts

# Input listener to trigger message sending events
def input_listener(node, message_trigger_queue):
    while True:
        input("Press Enter to trigger a message sending event...\n")
        if not node.message_queue.empty():
            message = node.message_queue.get()
            message_trigger_queue.put(message)

# Simulate clock synchronization before the experiment starts
# synchronize_clocks()

# Create two nodes (for simulation purposes)
node_0 = Node(0, ["msg1", "msg2"])
# node_1 = Node(1, ["msg3", "msg4"])

# Message trigger queue (shared between the input listener and sender thread)
message_trigger_queue_0 = Queue()

# Start the sender thread (Node 0 sending to Node 1)
sender_thread_0 = threading.Thread(target=sender_thread, args=(node_0, 1, message_trigger_queue_0), daemon=True)
sender_thread_0.start()

# Start the receiver thread for Node 1
receiver_thread_1 = threading.Thread(target=receiver_thread, args=(1,), daemon=True)
receiver_thread_1.start()

# Start the input listener (for user message triggers) in a separate thread
input_thread_0 = threading.Thread(target=input_listener, args=(node_0, message_trigger_queue_0), daemon=True)
input_thread_0.start()

# Keep the main thread alive
while True:
    time.sleep(1)
