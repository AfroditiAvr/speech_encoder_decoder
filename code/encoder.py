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


def RPE_frame_st_coder(s0: np.ndarray, prev_frame_st_resd: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    processed = preprocess(s0)

    pol_coef = solve_predictor_coefficients(processed, order=9)
    ref_coef = polynomial_coeff_to_reflection_coeff(pol_coef, 0)
    LAR = reflection_coef_to_LAR(ref_coef)
    LARc = quantization_coding_LAR(LAR)

    # Αποκωδικοποίηση των LARc
    dec_LAR = decode_of_LARc(LARc)
    dec_r = LAR_to_reflection_coef(dec_LAR)

    # Debugging print
    result = reflection_coeff_to_polynomial_coeff(dec_r)

    # Διόρθωση αποθήκευσης πολυωνυμικών συντελεστών
    if isinstance(result, (list, tuple)):
        dec_pol = np.array(result[0])
    else:
        dec_pol = np.array(result)

    # Υπολογισμός συντελεστών φίλτρου
    filter_coeffs = np.concatenate(([dec_pol[0]], -dec_pol[1:]))

    curr_frame_st_resd = lfilter([1], filter_coeffs, processed)

    return dec_LAR, curr_frame_st_resd


def long_term_analysis_filtering(
        d: np.ndarray,
        prev_d_reconstructed: np.ndarray,
        ltp_params: list
) -> Tuple[np.ndarray, np.ndarray]:
      #LTP analysis filtering 

    e = np.zeros(160, dtype=np.float32)
    current_d_reconstructed = np.zeros(160, dtype=np.float32)

    # Επεξεργασία κάθε subframe (j=0 to 3)
    for j in range(4):
        Nj, bj = ltp_params[j]  # Παίρνω τις LTP παραμέτρους για αυτό το subframe

        for k in range(40):
            global_index = j * 40 + k  # Global θέση στο frame 160 δειγμάτων
            lagged_index = global_index - Nj  # Θέση στο προηγούμενο buffer

            # σωστό indexing για το buffer του προηγούμενου residual
            if lagged_index < 0:
                d_star_shifted = prev_d_reconstructed[lagged_index + len(prev_d_reconstructed)]
            else:
                d_star_shifted = prev_d_reconstructed[lagged_index] if lagged_index < len(prev_d_reconstructed) else 0.0

            # υπολογισμός των προβλεπόμενων signal και residual
            d_estimated = bj * d_star_shifted
            e[global_index] = d[global_index] - d_estimated
            current_d_reconstructed[global_index] = e[global_index] + d_estimated

    # Ενημέρωση buffer: κρατάει τα τελευταία  120 δείγματα (prev_d + current_d)
    updated_prev_d = np.roll(prev_d_reconstructed, -40)
    updated_prev_d[-40:] = current_d_reconstructed[-40:]

    return e, updated_prev_d


def long_term_synthesis_filtering(
        e: np.ndarray,
        prev_d_reconstructed: np.ndarray,
        ltp_params: list
) -> Tuple[np.ndarray, np.ndarray]:
    #Long-Term Synthesis Filtering

    d_reconstructed = np.zeros(160, dtype=np.float32)

    for j in range(4):
        N_dec, b_dec = ltp_params[j]

        for k in range(40):
            global_index = j * 40 + k
            lagged_index = global_index - N_dec

            # σωστός χειρισμός των αρνητικών δεικτών
            if lagged_index < 0:
                d_double = prev_d_reconstructed[lagged_index + len(prev_d_reconstructed)]
            else:
                d_double = prev_d_reconstructed[lagged_index] if lagged_index < len(prev_d_reconstructed) else 0.0

            # υπολογισμός d'(n)
            d_reconstructed[global_index] = e[global_index] + b_dec * d_double

    # διατήρηση των τελευταίων 120 δειγμάτων
    updated_prev_d = np.roll(prev_d_reconstructed, -40)
    updated_prev_d[-40:] = d_reconstructed[-40:]

    return d_reconstructed, updated_prev_d

def RPE_frame_slt_coder(
        s0: np.ndarray,
        prev_frame_st_resd: np.ndarray
) -> Tuple[np.ndarray, List[int], List[int], np.ndarray, np.ndarray]:
    """
    Implements the RPE frame selection and LTP analysis.
    """

    # Split the signal into subframes
    subframes = split_into_subframes(s0)
    Nc, bc = [], []
    ltp_params = []
    ltp_params_en = []

    # Buffers for reconstructed residuals and excitation
    curr_frame_ex_full = np.zeros(160, dtype=np.float32)
    curr_frame_st_resd = np.zeros(160, dtype=np.float32)

    # Process each subframe (j=0 to 3)
    for j, subframe in enumerate(subframes):
        # Compute LTP parameters for the subframe
        Nj, bj = RPE_subframe_slt_lte(subframe, prev_frame_st_resd)

        # Encode LTP parameters
        Ncj = encode_ltp_lag(Nj)
        bcj = encode_gain(bj)

        # Decode LTP parameters
        Nj_dec = decode_ltp_lag(Ncj)
        bj_dec = decode_gain(bcj)

        # Store decoded values
        ltp_params.append((Nj_dec, bj_dec))
        ltp_params_en.append((Ncj, bcj))


        Nc.append(Ncj)
        bc.append(bcj)

    # Apply long-term prediction analysis filtering
    curr_frame_ex_full, updated_prev_residual = long_term_analysis_filtering(
        s0, prev_frame_st_resd, ltp_params
    )

    # Apply long-term synthesis filtering
    curr_frame_st_resd, updated_prev_residual = long_term_synthesis_filtering(
        curr_frame_ex_full, updated_prev_residual, ltp_params
    )

    # Placeholder for LAR coefficients
    LARc = np.zeros(8, dtype=np.float32)
    LARc, curr_frame_st_resd = RPE_frame_st_coder(curr_frame_st_resd, prev_frame_st_resd)

    return LARc, Nc, bc, curr_frame_ex_full, curr_frame_st_resd
