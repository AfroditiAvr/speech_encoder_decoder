from typing import Tuple, Optional

import numpy as np

#Οι συναρτήσεις reflection_coeff_to_polynomial_coeff, polynomial_coeff_to_reflection_coeff, _levup και _levdown μας δόθηκαν υλοποιημένες
#Οι υπόλοιπες συναρτήσεις υλοποιήθηκαν απο εμάς

def reflection_coeff_to_polynomial_coeff(kr: np.ndarray):
        """
        Converts the reflection coefficients `r` to polynomial coefficients `a`

        :param kr: (np.array) the vector containing the reflection coefficients

        :return: (np.array) the vector of polynomial coefficients,
                (float) the final prediction error, e_final, based on the zero lag autocorrelation, R0 (default: 0.).
        """
        
        
        # p is the order of the prediction polynomial.
        p = kr.size
        # set a to be an actual polynomial
        a = np.array([1.0, kr[0]])
        # a (p)-size vector
        e = np.zeros(shape=(p,))
        
        # Set the e0 parameter equal to 0., by default
        e0 = 0.
        
        # Initial value
        e[0] = e0 * (1 - np.conj(kr[0]) * kr[0])
        
        # Recursive steps
        for k in range(1, p):
                a_, e_k_ = _levup(a, kr[k], e[k-1])
                
                a = a_
                e[k] = e_k_
        
        e_final = e[-1]

        
        return a, e_final



def polynomial_coeff_to_reflection_coeff(
        a: np.ndarray, 
        e_final: float = 0.
        ) -> np.ndarray:
        """
        Converts the polynomial coefficients `a` to the reflection coefficients `r`.
        If a[0] != 1, then the function normalizes the prediction polynomial by a[0]

        :param a: (np.ndarray) the vector containing the polynomial prediction coefficients
        :param e_final: (float) the final prediction error (default: 0.0)

        :return: (np.array) the reflection coefficients `r`.
        """

        if a.size <= 1:
                return np.array([])
        
        if a[0] == 0.:
                raise ValueError("Leading coefficient cannot be zero.")
        
        # Normalize by a[0]
        a = a / a[0]
        
        # The leading one does not count
        p  = a.size - 1
        e  = np.zeros(shape=(p,))
        kr = np.zeros(shape=(p,))
        
        e[-1]  = e_final
        kr[-1] = a[-1]
        
        for k in np.arange(p-2, -1, -1):
                a, e_k = _levdown(a, e[k+1])
                
                e[k]  = e_k
                kr[k] = a[-1]


        
        return kr

      

def _levup(acur: np.ndarray, knxt: np.ndarray, ecur: float):
        
        # Drop the leading 1, it is not needed in the stepup
        acur = acur[1:]
        
        # Matrix formulation from Stoica is used to avoid looping
        acur_0     = np.append(arr=acur,       values=[0])
        acur_rev_1 = np.append(arr=acur[::-1], values=[1.])
        
        anxt = acur_0 + knxt * np.conj(acur_rev_1)
        
        enxt = (1.0 - np.dot(np.conj(knxt), knxt)) * ecur
        
        # Insert '1' at the beginning to make it an actual polynomial
        anxt = np.insert(anxt, 0, 1.0)
        
        return anxt, enxt


def _levdown(anxt: np.ndarray, enxt: Optional[float] = None) -> Tuple[np.ndarray, float]:
        
        
        # Drop the leading 1 (not needed in the step-down)
        anxt = anxt[1:]
        
        # Extract the (k+1)-th reflection coefficient
        knxt = anxt[-1]
        
        if knxt == 1.0:
                raise ValueError("At least one of the reflection coefficients is equal to one.\nThe algorithm fails for this case.")
        
        # A matrix formulation from Stoica is used to avoid looping
        acur = (anxt[:-1] - knxt * np.conj(anxt[::-1][1:])) / (1 - np.abs(knxt) ** 2)
                
        ecur = enxt / (1 - np.dot(np.conj(knxt).transpose(), knxt) ) if enxt is not None else None
        
        # Insert the constant 1 coefficient to make it a true polynomial
        acur = np.insert(acur, 0, 1)
        
        return acur, ecur

##################################################################
##################################################################
##################################################################
##################################################################

# Συντελεστές από το πρότυπο
ALPHA = 32735 * 2**-15  # Offset Compensation
BETA = 28180 * 2**-15   # Preemphasis

def split_into_frames(signal, frame_size=160):
    """Χωρίζει το σήμα σε frames σταθερού μήκους."""
    num_frames = len(signal) // frame_size
    frames = [signal[i * frame_size: (i + 1) * frame_size] for i in range(num_frames)]
    return frames

def split_into_subframes(frame, subframe_size=40):
    # Χωρίζει ένα frame σε 4 subframes 40 δειγμάτων το καθένα
    num_subframes = len(frame) // subframe_size

    subframes = [frame[i * subframe_size: (i + 1) * subframe_size] for i in range(num_subframes)]
    return subframes


def compute_excitation_subsequences(x):
    subsequences = []
    powers = []

    # Υπολογισμός των 4 υποακολουθιών
    for j in range(4):
        subseq = x[j:j + 13 * 3:3]  # Σωστή εξαγωγή 13 στοιχείων

        subsequences.append(subseq)

        # Υπολογισμός ισχύος (Άθροισμα τετραγώνων των στοιχείων)
        power = sum(s ** 2 for s in subseq)
        powers.append(power)

    # Επιλογή της υποακολουθίας με τη μέγιστη ισχύ
    max_index = np.argmax(powers)
    best_subseq = subsequences[max_index]

    return best_subseq, max_index

def quantize_xmax(xmax):
    quantization_table = [
        (0, 32), (64, 33), (96, 34), (128, 35), (160, 36), (192, 37), (224, 38), (256, 39),
        (288, 40), (320, 41), (352, 42), (384, 43), (416, 44), (448, 45), (480, 46), (512, 47),
        (576, 48), (640, 49), (704, 50), (768, 51), (832, 52), (896, 53), (960, 54), (1024, 55),
        (1152, 56), (1280, 57), (1408, 58), (1536, 59), (1664, 60), (1792, 61), (1920, 62), (2048, 63)
    ]
    for lower_bound, xmc in quantization_table:
        if xmax < lower_bound:
            return xmc
    return 63

def dequantize_xmax(xmc):
    dequantization_table = {
        32: 2048, 33: 2304, 34: 2595, 35: 3072, 36: 3328, 37: 3584, 38: 4096, 39: 4608,
        40: 5120, 41: 5632, 42: 6144, 43: 6656, 44: 7168, 45: 7680, 46: 8192, 47: 9216,
        48: 10240, 49: 11264, 50: 12288, 51: 13312, 52: 14336, 53: 15360, 54: 16384,
        55: 18432, 56: 20480, 57: 22528, 58: 24576, 59: 26624, 60: 28672, 61: 30719,
        62: 32767, 63: 32767
    }
    return dequantization_table.get(xmc, 32767)


def compute_x_prime(x_M, x_max_quantized):
    """
    Υπολογίζει τα 13 x'(i) = x_M(i) / x'_max
    όπου x'_max είναι η αποκβαντισμένη εκδοχή της x_max.

    Παράμετροι:
    - x_M: Λίστα με 13 δείγματα της επιλεγμένης υπακολουθίας x_M(i)
    - x_max_quantized: Η αποκβαντισμένη τιμή του x_max

    Επιστρέφει:
    - Λίστα με 13 κανονικοποιημένα δείγματα x'(i)
    """
    if x_max_quantized == 0:
        raise ValueError("Η αποκβαντισμένη x_max δεν μπορεί να είναι 0.")

    x_prime = [x / x_max_quantized for x in x_M]
    return x_prime


def quantize_rpe_samples(x_normalized):

    # Μετατροπή σε τύπο float32 αν δεν είναι ήδη
    x_normalized = np.asarray(x_normalized, dtype=np.float32)

    x_normalized = x_normalized * (2**15)   # Η πράξη αυτή ΔΕΝ αλλάζει το μέγεθος


    # Επίπεδα
    levels = np.array([-32768, -24576, -16384, -8192, 0, 8192, 16384, 24576, 32768], dtype=np.float32)

    # Χρησιμοποιούμε np.digitize για να κβαντίσουμε τα δεδομένα
    indices = np.digitize(x_normalized, levels, right=False) - 1  # Κάνουμε right=False αν χρειάζεται

    # Ελέγχουμε τα indices
    indices = np.clip(indices, 0, len(levels) - 2)  # Εξασφαλίζουμε ότι οι δείκτες είναι μέσα στα όρια


    return indices

def dequantize_rpe_samples(xmc):
    levels = np.array([-28672, -20480, -12288, -4096, 0, 4096, 12288, 20480, 28672])
    levels = levels/(2**15)
    return levels[xmc]

def compute_xM_approximations(xc_prime, x_max_prime):
    result = xc_prime * x_max_prime
    return result




def insert_zeros(xM_prime, Mc):
    """
    Τοποθετεί τα στοιχεία της επιλεγμένης υποακολουθίας στη σωστή θέση, αφήνοντας ακριβώς 3 μηδενικά ενδιάμεσα.

    :param xM_prime: Η υποακολουθία των 13 τιμών που επιλέχθηκε.
    :param Mc: Η θέση έναρξης της υποακολουθίας.
    :return: Ένας πίνακας 40 στοιχείων όπου μόνο οι θέσεις που ανήκουν στην επιλεγμένη υποακολουθία περιέχουν τιμές.
    """
    e_prime = np.zeros(40)  # Δημιουργία πίνακα 40 μηδενικών
    index = Mc  # Ξεκινάμε από το Mc

    for i in range(len(xM_prime)):
        if index >= 40:
            break  # Αν φτάσουμε στο τέλος του πίνακα, σταματάμε
        e_prime[index] = xM_prime[i]  # Τοποθετούμε την τιμή
        index += 4  # Προχωράμε 4 θέσεις (3 μηδενικά ανάμεσα)

    return e_prime


def weighting_filter(e: np.ndarray) -> np.ndarray:
    """
    Applies the FIR 'block filter' algorithm to a sequence of 40 samples using the impulse response H(i) given in the image.

    :param e: Input sequence of 40 samples (numpy array of shape (40,))
    :return: Filtered sequence of 40 samples (numpy array of shape (40,))
    """
    assert len(e) == 40, "Input sequence must have exactly 40 samples"

    # Impulse response H(i) scaled by 2^13 as given in the table
    H = np.array([-134, -374, 0, 2054, 5741, 8192, 5741, 2054,  0, -374, -134]) / (2 ** 13)

    # Initialize output sequence
    x = np.zeros_like(e)

    gain = 2.779

    # Apply the filtering according to equation (3.20)
    for k in range(40):
        for i in range(11):  # H(i) is nonzero only for i = 0 to 10
            index = k + 5 - i
            if 0 <= index < 40:
                x[k] += H[i] * e[index]
            else:
                x[k] += H[i] * 0  # Consider out-of-bounds values as 0

    x *= gain

    return x



def preprocess(frame):
    """Επεξεργάζεται ένα frame εφαρμόζοντας αντιστάθμιση μετατόπισης (Offset Compensation)."""
    sof = np.zeros_like(frame)
    for k in range(len(frame)):
        if k == 0:
            sof[k] = frame[k]
        else:
            sof[k] = frame[k] - frame[k - 1] + ALPHA * sof[k - 1]

    s = np.zeros_like(sof)
    for k in range(len(sof)):
        if k == 0:
            s[k] = sof[k]
        else:
            s[k] = sof[k] - BETA * sof[k - 1]


    return s


def autocorrelation(frame, lag):
    acf = 0
    for i in range(lag, len(frame)):
        acf += frame[i] * frame[i - lag]
    return acf

def compute_R_and_r(signal, order=9):
    """
    Υπολογίζει τον πίνακα R και το διάνυσμα r για το σύστημα κανονικών εξισώσεων.
    """

    # Υπολογισμός του πίνακα R (8x8)
    R = np.zeros((order, order))
    for i in range(order):
        for j in range(order):
            R[i, j] = autocorrelation(signal, abs(i - j))


    r = np.zeros(order)
    for i in range(order):  # Ξεκινά από 1 αντί για 0
        r[i] = autocorrelation(signal, i )

    return R, r


def solve_predictor_coefficients(signal, order=9):
    """
    Λύνει τις κανονικές εξισώσεις Rw = r για να υπολογίσει τους συντελεστές πρόβλεψης.
    """
    # Υπολογισμός R και r
    R, r = compute_R_and_r(signal, order)

    # Επίλυση συστήματος Rw = r
    w = np.linalg.solve(R, r)

    return w

def reflection_coef_to_LAR(r):

    LAR = np.zeros_like(r, dtype=np.float64)
    for i in range(len(r)):
        abs_r = abs(r[i])
        if abs_r < 0.675:
            LAR[i] = r[i]
        elif abs_r < 0.950:
            LAR[i] = np.sign(r[i]) * (2 * abs_r - 0.675)
        else:
            LAR[i] = np.sign(r[i]) * (8 * abs_r - 6.375)
    return LAR


def LAR_to_reflection_coef(LAR):
    r = np.zeros_like(LAR, dtype=np.float64)
    for i in range(len(LAR)):
        abs_LAR = abs(LAR[i])
        if abs_LAR < 0.675:
            r[i] = LAR[i]
        elif abs_LAR < 1.225:
            r[i] = np.sign(LAR[i]) * (0.5 * abs_LAR + 0.3375)
        else:
            r[i] = np.sign(LAR[i]) * (0.125 * abs_LAR + 0.796875)

    return r




def quantization_coding_LAR(LAR):
    z = np.zeros_like(LAR, dtype=float)  # Αποθήκευση ενδιάμεσων υπολογισμών
    LARc = np.zeros_like(LAR, dtype=float)  # Ακέραιες τιμές μετά την κβάντιση
    A = [20, 20, 20, 20, 13.637, 15, 8.334, 8.824]
    B = [0, 0, 4, -5, 0.184, -3.5, -0.666, -2.235]
    LARc_min = [-32, -32, -16, -16, -8, -8, -4, -4]
    LARc_max = [31, 31, 15, 15, 7, 7, 3, 3]
    for i in range(len(LAR)):
        z[i] = A[i] * LAR[i] + B[i]
        LARc[i] = int(z[i] + np.sign(z[i]) * 0.5)
        if LARc[i]<LARc_min[i]:
            LARc[i] = LARc_min[i]
        if LARc[i]>LARc_max[i]:
            LARc[i] = LARc_max[i]

    return LARc


def decode_of_LARc(LARc):
    A = [20, 20, 20, 20, 13.637, 15, 8.334, 8.824]
    B = [0, 0, 4, -5, 0.184, -3.5, -0.666, -2.235]
    dec_LAR = np.zeros_like(LARc, dtype=float)


    for i in range(len(LARc)):
        dec_LAR[i] = (LARc[i] - B[i]) / A[i]

    return dec_LAR


def iir_deemphasis(sr):
    """
    Υλοποιεί το IIR-deemphasis φίλτρο.
    sr: Λίστα με τα δείγματα εισόδου.
    beta: Συντελεστής του φίλτρου.
    Επιστρέφει: Λίστα με τα φιλτραρισμένα δείγματα.
    """
    sro = [0] * len(sr)  # Αρχικοποίηση εξόδου με μηδενικές τιμές

    if len(sr) > 0:
        sro[0] = sr[0]  # Αρχική συνθήκη

    for k in range(1, len(sr)):
        sro[k] = sr[k] + BETA * sro[k - 1]


    return sro


def compute_cross_correlation(subframe, prev_residual, lag_range=(40, 120)):
    """
    Υπολογίζει τη συσχέτιση Rj(λ) μεταξύ του τρέχοντος subframe και των προηγούμενων
    """
    min_lag, max_lag = lag_range
    cross_correlation = np.zeros(max_lag - min_lag + 1)

    for idx, lag in enumerate(range(min_lag, max_lag + 1)):
        cross_correlation[idx] = np.sum(
            subframe * prev_residual[-lag - len(subframe):-lag]
        )

    return cross_correlation


def find_optimal_lag(cross_correlation, lag_range=(40, 120)):
    """
    Βρίσκει το βέλτιστο Νj όπου το Rj(λ) μεγιστοποιείται.
    """
    min_lag, _ = lag_range
    optimal_index = np.argmax(cross_correlation)
    optimal_lag = min_lag + optimal_index
    return optimal_lag


def compute_gain_factor(subframe, prev_residual, optimal_lag):
    """
    Υπολογίζει τον συντελεστή ενίσχυσης bj.
    """
    numerator = np.sum(subframe * prev_residual[-(optimal_lag + len(subframe)):-optimal_lag])
    denominator = np.sum(prev_residual[-(optimal_lag + len(subframe)):-optimal_lag] ** 2)

    if denominator == 0:
        return 0  # Αποφυγή διαίρεσης με το μηδέν

    bj = numerator / denominator
    return bj


from typing import Tuple


def RPE_subframe_slt_lte(d: np.ndarray, prev_d: np.ndarray) -> Tuple[int, float]:
    """
    Estimates LTP parameters (pitch period N and gain factor b) for a subframe.
    """
    # Compute cross-correlation between subframe and previous residual
    cross_corr = compute_cross_correlation(subframe=d, prev_residual=prev_d)

    # Find optimal lag (pitch period N)
    N = find_optimal_lag(cross_correlation=cross_corr)

    # Calculate gain factor b using the optimal lag
    b = compute_gain_factor(subframe=d, prev_residual=prev_d, optimal_lag=N)

    return N, b


def encode_ltp_lag(Nj: int) -> int:
    """Encodes the LTP lag Nj (40 ≤ Nj ≤ 120) into a 7-bit integer Ncj."""
    # Clip to valid range and cast to integer
    Ncj = int(np.clip(Nj, 40, 120))
    return Ncj


def decode_ltp_lag(Ncj: int) -> int:
    """Decodes the 7-bit integer Ncj into the LTP lag Nj' (40 ≤ Nj' ≤ 120)."""
    # Directly return Ncj (assumes no transmission errors)
    Nj_dec = Ncj
    return Nj_dec


def encode_gain(bj):
    DLB = [0.2, 0.5, 0.8]
    if bj <= DLB[0]:
        bcj = 0
    elif DLB[0] < bj <= DLB[1]:
        bcj = 1
    elif DLB[1] < bj <= DLB[2]:
        bcj = 2
    else:
        bcj = 3
    return bcj


def decode_gain(bcj):
    QLB = [0.10, 0.35, 0.65, 1.00]
    bj_dec = QLB[bcj]
    return bj_dec


