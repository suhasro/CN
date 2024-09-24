import threading
import time
import random
from queue import Queue

# Shared resource to simulate the busy state of the channel
channel_busy = threading.Lock()  # Lock represents whether the channel is busy or not

# Function to simulate clock synchronization (e.g., via NTP)
def synchronize_clocks():
    print("Clock synchronized.")

# Node class definition
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

    def send_rts(self, destination):
        if channel_busy.locked():
            print(f"Node {self.node_id}: Channel is busy, cannot send RTS.")
            return False
        
        rts_message = self.create_mac_header('00', self.node_id, destination.node_id)
        print(f"Node {self.node_id}: Sending RTS to Node {destination.node_id}. RTS message: {rts_message}")
        return True

    def receive_rts(self, source):
        print(f"Node {self.node_id}: Received RTS from Node {source.node_id}. Checking for collision...")
        collision_occurred = random.choice([True, False])  # Simulate collision detection
        if collision_occurred:
            print(f"Node {self.node_id}: Collision detected! Not sending CTS.")
            return False
        else:
            print(f"Node {self.node_id}: No collision detected. Sending CTS.")
            self.send_cts(source)
            return True

    def send_cts(self, source):
        cts_message = self.create_mac_header('01', self.node_id, source.node_id)
        print(f"Node {self.node_id}: Sending CTS to Node {source.node_id}. CTS message: {cts_message}")
        
    def receive_cts(self, source):
        print(f"Node {self.node_id}: Listening for CTS from Node {source.node_id}...")
        # Simulate receiving CTS
        cts_received = random.choice([True, False])
        if cts_received:
            print(f"Node {self.node_id}: Received CTS from Node {source.node_id}.")
            return True
        else:
            print(f"Node {self.node_id}: No CTS received. Collision likely occurred.")
            return False

    def send_data(self, destination, message):
        data_message = self.create_mac_header('10', self.node_id, destination.node_id, message)
        print(f"Node {self.node_id}: Sending data '{message}' to Node {destination.node_id}. Data message: {data_message}")
        destination.receive_data(self, message)

    def receive_data(self, source, message):
        print(f"Node {self.node_id}: Received data '{message}' from Node {source.node_id}. Sending ACK.")
        data_received = random.choice([True, False])
        if data_received:
            print(f"Node {self.node_id}: ACK received from Node {source.node_id}.")
            return True
        else:
            print(f"Node {self.node_id}: No ACK received. Collision likely occurred.")
            return False

    def send_ack(self, source):
        ack_message = self.create_mac_header('11', self.node_id, source.node_id)
        print(f"Node {self.node_id}: Sending ACK to Node {source.node_id}. ACK message: {ack_message}")
        
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
synchronize_clocks()

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