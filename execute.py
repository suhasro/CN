#INSTRUCTIONS : ENTER THE INPUT STRING CONTAINING 0's AND 1's,ENTER TWO REAL NUMBERS IN THE RANGE(0,1) AND RUN THE CODE.
import pyaudio
import threading
import time
import numpy as np
import math
import random
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

def receive_signal(): # This function is extracted from ChatGpt
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=FS, input=True, frames_per_buffer=CHUNK)

    print("Receiving signal...")
    received_bits = []

    try:
        while True:
            data = stream.read(CHUNK)

            signal = np.frombuffer(data, dtype=np.float32)
            
            detected_freq = detect_frequency(signal)
            
            bit = frequency_to_bit(detected_freq)
            if bit != None:
                received_bits.append(bit)
            print(f"Received bit: {bit}, {detected_freq}")
    
    except KeyboardInterrupt:
        print("\nStopped receiving.")
    
    finally:
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

messages = []
numbers = []

for _ in range(2):
    line = input("Enter a message and a number: ")
    msg, num = line.rsplit(' ', 1)
    messages.append(msg)
    numbers.append(int(num))

print(messages)

while True:
    final = receive_signal()
    if final:
        time.sleep(3)
    else:
        bit_sequence = "####" + messages[0] + "####"

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