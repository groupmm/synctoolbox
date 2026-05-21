import librosa
import numpy as np
from scipy import signal
from scipy.interpolate import interp1d
from typing import Tuple


def smooth_downsample_feature(f_feature: np.ndarray,
                              input_feature_rate: float,
                              win_len_smooth: int = 0,
                              downsamp_smooth: int = 1) -> Tuple[np.ndarray, float]:
    """Temporal smoothing and downsampling of a feature sequence

    Parameters
    ----------
    f_feature : np.ndarray
        Input feature sequence, size dxN

    input_feature_rate : float
        Input feature rate in Hz

    win_len_smooth : int
        Smoothing window length. For 0, no smoothing is applied.

    downsamp_smooth : int
        Downsampling factor. For 1, no downsampling is applied.

    Returns
    -------
    f_feature_stat : np.ndarray
        Downsampled & smoothed feature.

    new_feature_rate : float
        New feature rate after downsampling
    """
    if win_len_smooth != 0 or downsamp_smooth != 1:
        # hack to get the same results as on MATLAB
        stat_window = np.hanning(win_len_smooth+2)[1:-1]
        stat_window /= np.sum(stat_window)

        # upfirdn filters and downsamples each column of f_stat_help
        f_feature_stat = signal.upfirdn(h=stat_window, x=f_feature, up=1, down=downsamp_smooth)
        seg_num = f_feature.shape[1]
        stat_num = int(np.ceil(seg_num / downsamp_smooth))
        cut = int(np.floor((win_len_smooth - 1) / (2 * downsamp_smooth)))
        f_feature_stat = f_feature_stat[:, cut: stat_num + cut]
    else:
        f_feature_stat = f_feature

    new_feature_rate = input_feature_rate / downsamp_smooth

    return f_feature_stat, new_feature_rate


def normalize_feature(feature: np.ndarray,
                      norm_ord: int,
                      threshold: float) -> np.ndarray:
    """Normalizes a feature sequence according to the l^norm_ord norm.

    Parameters
    ----------
    feature : np.ndarray
        Input feature sequence of size d x N
            d: dimensionality of feature vectors
            N: number of feature vectors (time in frames)

    norm_ord : int
        Norm degree

    threshold : float
        If the norm falls below threshold for a feature vector, then the
        normalized feature vector is set to be the normalized unit vector.

    Returns
    -------
    f_normalized : np.ndarray
        Normalized feature sequence
    """
    d, N = feature.shape

    # normalize the vectors according to the l^norm_ord norm
    unit_vec = np.ones(d)
    unit_vec = unit_vec / np.linalg.norm(unit_vec, norm_ord)

    norms = np.linalg.norm(feature, ord=norm_ord, axis=0)
    below_threshold = norms < threshold
    safe_norms = np.where(below_threshold, 1.0, norms)
    f_normalized = feature / safe_norms.reshape(1, N)
    f_normalized[:, below_threshold] = unit_vec.reshape(d, 1)

    return f_normalized


def estimate_tuning(x: np.ndarray,
                    Fs: float,
                    N: int = 16384,
                    gamma: float = 100,
                    local: bool = True,
                    filt: bool = True,
                    filt_len: int = 101) -> int:
    """Compute tuning deviation in cents for an audio signal.

    Parameters
    ----------
    x : np.ndarray
        Input signal

    Fs : float
        Sampling rate

    N : int
        Window size

    gamma : float
        Constant for logarithmic compression

    local : bool
        If `True`, computes STFT and averages; otherwise computes global DFT

    filt : bool
        If `True`, applies local frequency averaging and by rectification

    filt_len : int
        Filter length for local frequency averaging (length given in cents)

    Returns
    -------
    tuning : int
        Estimated tuning deviation for ``x`` (in cents)
    """
    # TODO supply N in seconds and compute window size in frames via Fs
    v, _ = __compute_freq_distribution(x, Fs, N, gamma, local, filt, filt_len)
    _, _, _, tuning, _ = __tuning_similarity(v)
    return tuning


def __compute_freq_distribution(x, Fs, N=16384, gamma=100.0, local=True, filt=True, filt_len=101):
    """Compute an overall frequency distribution for tuning estimation."""
    if local:
        if N > len(x) // 2:
            raise Exception('The signal length (%d) should be twice as long as the window length (%d)' % (len(x), N))
        Y = np.abs(librosa.stft(x, n_fft=N, hop_length=N // 2, win_length=N,
                                window='hann', pad_mode='constant', center=True)) ** 2
        if gamma > 0:
            Y = np.log(1 + gamma * Y)
        Y = np.sum(Y, axis=1)
        F_coef = librosa.fft_frequencies(sr=Fs, n_fft=N)
    else:
        N = len(x)
        Y = np.abs(np.fft.fft(x)) / Fs
        Y = Y[:N // 2 + 1]
        Y = np.log(1 + gamma * Y)
        F_coef = np.arange(N // 2 + 1).astype(float) * Fs / N

    f_pitch = lambda p: 440 * 2 ** ((p - 69) / 12)
    F_min = f_pitch(24)
    F_max = f_pitch(108)
    F_coef_log, F_coef_cents = __compute_f_coef_log(R=1, F_min=F_min, F_max=F_max)
    Y_int = interp1d(F_coef, Y, kind='cubic', fill_value='extrapolate')(F_coef_log)
    v = Y_int / np.max(Y_int)

    if filt:
        filt_kernel = np.ones(filt_len)
        Y_smooth = signal.convolve(Y_int, filt_kernel, mode='same') / filt_len
        Y_rectified = Y_int - Y_smooth
        Y_rectified[Y_rectified < 0] = 0
        v = Y_rectified / np.max(Y_rectified)

    return v, F_coef_cents


def __compute_f_coef_log(R, F_min, F_max):
    n_bins = np.ceil(1200 * np.log2(F_max / F_min) / R).astype(int)
    F_coef_log = 2 ** (np.arange(0, n_bins) * R / 1200) * F_min
    F_coef_cents = 1200 * np.log2(F_coef_log / F_min)
    return F_coef_log, F_coef_cents


def __tuning_similarity(v):
    theta_axis = np.arange(-50, 50)
    num_theta = len(theta_axis)
    sim = np.zeros(num_theta)
    M = len(v)
    for i in range(num_theta):
        theta = theta_axis[i]
        template = __template_comb(M=M, theta=theta)
        sim[i] = np.inner(template, v)
    sim = sim / np.max(sim)
    ind_max = np.argmax(sim)
    theta_max = theta_axis[ind_max]
    template_max = __template_comb(M=M, theta=theta_max)
    return theta_axis, sim, ind_max, theta_max, template_max


def __template_comb(M, theta=0):
    template = np.zeros(M)
    peak_positions = (np.arange(0, M, 100) + theta)
    peak_positions = np.intersect1d(peak_positions, np.arange(M)).astype(int)
    template[peak_positions] = 1
    return template


def shift_chroma_vectors(chroma: np.ndarray,
                         chroma_shift: int) -> np.ndarray:
    """Shift chroma representation by the given number of semitones.
    Format is assumed to be 12xN

    Parameters
    ----------
    chroma: np.ndarray [shape=(12, N)]
        Chroma representation

    chroma_shift: int
        Chroma shift

    Returns
    -------
    shifted_chroma: np.ndarray
        Shifted chroma representation
    """
    shifted_chroma = np.roll(chroma, chroma_shift, axis=0)
    return shifted_chroma
