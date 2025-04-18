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
    # Read the audio file
    samplerate, data = wavfile.read("ena_dio_tria.wav")

    # Convert to mono if stereo
    if len(data.shape) > 1:
        data = data[:, 0]

    # Χωρισμός σε frames
    frame_size = 160
    num_frames = len(data) // frame_size
    frames = [data[i * frame_size: (i + 1) * frame_size] for i in range(num_frames)]

    # List for decoded signal
    decoded_signal = []
    prev_residual = np.zeros(frame_size)

    for frame in frames:
        # Encoding
        LARc, Nc, bc, curr_frame_ex_full, curr_residual = RPE_frame_slt_coder(frame, prev_residual)

        # Decoding
        decoded_frame, prev_residual = RPE_frame_slt_decoder(LARc, Nc, bc, curr_frame_ex_full, prev_residual)

        # Αποθήκευση αποτελεσμάτων
        decoded_signal.extend(decoded_frame)

    #clipping
    decoded_signal = np.clip(decoded_signal, -32768, 32767)
    decoded_signal = np.array(decoded_signal, dtype=np.int16)

    # Save the decoded audio file
    wavfile.write("decoded_speech.wav", samplerate, decoded_signal)

    # Plot signals
    plt.figure(figsize=(10, 5))
    plt.subplot(2, 1, 1)
    plt.plot(data[:len(decoded_signal)], label="Original Signal")
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(decoded_signal, label="Reconstructed Signal", color='orange')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
