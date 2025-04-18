import numpy as np
from hw_utils import polynomial_coeff_to_reflection_coeff, reflection_coeff_to_polynomial_coeff
from scipy.signal import lfilter
import matplotlib.pyplot as plt
from scipy.io import wavfile
from typing import List, Tuple
from hw_utils import *
from encoder import *
from decoder import *


# Διαβάζουμε το αρχείο .wav
samplerate, data = wavfile.read("ena_dio_tria.wav")

# Εμφανίζουμε κάποιες πληροφορίες
print(f"Sample Rate: {samplerate} Hz")
print(f"Data Shape: {data.shape}")
print(f"Data Type: {data.dtype}")

# Συντελεστές από το πρότυπο
ALPHA = 32735 * 2**-15  # Offset Compensation
BETA = 28180 * 2**-15   # Preemphasis

def main():
    # Διαβάζουμε το αρχείο ήχου
    samplerate, data = wavfile.read("ena_dio_tria.wav")

    # Αν το σήμα είναι στερεοφωνικό, το μετατρέπουμε σε μονοφωνικό
    if len(data.shape) > 1:
        data = data[:, 0]  # Παίρνουμε μόνο το πρώτο κανάλι

    # Διαχωρισμός σε πλαίσια των 160 δειγμάτων
    frame_size = 160
    num_frames = len(data) // frame_size
    frames = [data[i * frame_size: (i + 1) * frame_size] for i in range(num_frames)]

    # Λίστες για αποθήκευση των κωδικοποιημένων δεδομένων
    LARc_list = []
    residuals_list = []
    prev_residual = np.zeros(frame_size)  # Αρχικοποίηση προηγούμενου residue

    # **1ο στάδιο: Κωδικοποίηση**
    for frame in frames:
        LARc, curr_residual = RPE_frame_st_coder(frame, prev_residual)
        LARc_list.append(LARc)
        residuals_list.append(curr_residual)
        prev_residual = curr_residual  # Ενημέρωση του προηγούμενου residue

    # **2ο στάδιο: Αποκωδικοποίηση**
    decoded_signal = []
    prev_residual = np.zeros(frame_size)  # Ξαναρχικοποιούμε το residue

    for i in range(num_frames):
        decoded_frame = PE_frame_st_decoder(LARc_list[i], residuals_list[i])
        decoded_signal.extend(decoded_frame)

    # Μετατροπή του αποκωδικοποιημένου σήματος σε int16 με περιορισμό
    decoded_signal = np.clip(decoded_signal, -32768, 32767)
    decoded_signal = np.array(decoded_signal, dtype=np.int16)

    # Αποθήκευση του αποκωδικοποιημένου αρχείου
    wavfile.write("decoded_speech.wav", samplerate, decoded_signal)

    # Σχεδίαση των σημάτων
    plt.figure(figsize=(10, 5))
    plt.subplot(2, 1, 1)
    plt.plot(data[:len(decoded_signal)], label="Αρχικό Σήμα")
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(decoded_signal, label="Ανακατασκευασμένο Σήμα", color='orange')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()


