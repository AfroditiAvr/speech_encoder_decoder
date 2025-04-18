import numpy as np
from hw_utils import polynomial_coeff_to_reflection_coeff, reflection_coeff_to_polynomial_coeff
from scipy.signal import lfilter
import matplotlib.pyplot as plt
from scipy.io import wavfile
from typing import List, Tuple
from hw_utils import *



# Διαβάζουμε το αρχείο .wav
samplerate, data = wavfile.read("ena_dio_tria.wav")

# Εμφανίζουμε κάποιες πληροφορίες
print(f"Sample Rate: {samplerate} Hz")
print(f"Data Shape: {data.shape}")
print(f"Data Type: {data.dtype}")

# Συντελεστές από το πρότυπο
ALPHA = 32735 * 2**-15  # Offset Compensation
BETA = 28180 * 2**-15   # Preemphasis



def PE_frame_st_decoder(LARc: np.ndarray, curr_frame_st_resd: np.ndarray):
    # Αποκωδικοποίηση των LARc
    dec_r = LAR_to_reflection_coef(LARc)

    # Debugging print για τον πίνακα dec_r

    # Διόρθωση αποθήκευσης των πολυωνυμικών συντελεστών
    result = reflection_coeff_to_polynomial_coeff(dec_r)

    if isinstance(result, (list, tuple)):
        dec_pol = np.array(result[0])
    else:
        dec_pol = np.array(result)


    # Έλεγχος για NaN ή Inf στους συντελεστές φίλτρου
    denominator = np.concatenate(([dec_pol[0]], -dec_pol[1:]))

    if np.any(np.isnan(denominator)) or np.any(np.isinf(denominator)):
        raise ValueError(f"Invalid filter coefficients: {denominator}")

    s0 = lfilter([1], denominator, curr_frame_st_resd)


    s0 = iir_deemphasis(s0)

    return s0



def RPE_frame_slt_decoder(
        LARc: np.ndarray,
        Nc: list,
        bc: list,
        curr_frame_ex_full: np.ndarray,
        prev_frame_st_resd: np.ndarray
):
    frame_size = 160
    subframe_size = 40
    num_subframes = 4

    # Αρχικοποίηση της εξόδου d′(n)
    curr_frame_st_resd = np.zeros(frame_size)

    for i in range(num_subframes):
        start = i * subframe_size
        end = start + subframe_size

        # Λήψη των τιμών Nc και bc για το συγκεκριμένο subframe
        N_prime = decode_ltp_lag(Nc[i])
        b_prime = decode_gain(bc[i])



        # Λήψη του excitation error e′(n)
        e_prime = curr_frame_ex_full[start:end]

        # Υπολογισμός του d′′(n)
        if start - N_prime >= 0:
            d_double_prime = curr_frame_st_resd[max(0, start - N_prime): start - N_prime + subframe_size]
        else:
            prev_start = frame_size + start - N_prime
            prev_end = prev_start + subframe_size
            if prev_start < 0:
                prev_start = 0  # Αποφυγή αρνητικών δεικτών
            d_double_prime = prev_frame_st_resd[prev_start: min(prev_end, frame_size)]

        # Προσαρμογή μήκους για αποφυγή σφαλμάτων
        d_double_prime = np.pad(d_double_prime, (0, subframe_size - len(d_double_prime)), mode='constant')

        # Υπολογισμός του d′(n)
        curr_frame_st_resd[start:end] = e_prime + b_prime * d_double_prime

    # Αποκωδικοποίηση μέσω PE_frame_st_decoder
    s0 = PE_frame_st_decoder(LARc, curr_frame_st_resd)

    return s0, curr_frame_st_resd
