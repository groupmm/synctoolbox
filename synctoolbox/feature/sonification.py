import numpy as np


def list_to_pitch_activations(note_list, num_frames, frame_rate):
    """Create a pitch activation matrix from a list of note events."""
    P = np.zeros((128, num_frames))
    F_coef_MIDI = np.arange(128) + 1
    for note in note_list:
        start_frame = max(0, int(note[0] * frame_rate))
        end_frame = min(num_frames, int((note[0] + note[1]) * frame_rate) + 1)
        P[int(note[2] - 1), start_frame:end_frame] = 1
    return P, F_coef_MIDI


def sonify_pitch_activations(P,
                             N,
                             frame_rate,
                             Fs,
                             min_pitch=1,
                             Fc=440,
                             harmonics_weights=(1,),
                             fading_msec=5):
    """Sonify a pitch activation matrix using sinusoidal tones."""
    fade_sample = int(fading_msec / 1000 * Fs)
    pitch_son = np.zeros((N,))

    for p in range(P.shape[0]):
        if np.sum(np.abs(P[p, :])) > 0:
            pitch = min_pitch + p
            freq = (2 ** ((pitch - 69) / 12)) * Fc
            sin_tone = np.zeros((N,))

            for i, cur_harmonic_weight in enumerate(harmonics_weights):
                sin_tone += cur_harmonic_weight * np.sin(2 * np.pi * (i + 1) * freq * np.arange(N) / Fs)

            weights = np.zeros((N,))
            for n in range(P.shape[1]):
                if np.abs(P[p, n]) > 0:
                    start = min(N, max(0, int((n - 0.5) * Fs / frame_rate)))
                    end = min(N, int((n + 0.5) * Fs / frame_rate))
                    fade_start = min(N, start + fade_sample)
                    fade_end = min(N, end + fade_sample)

                    weights[fade_start:end] += P[p, n]
                    weights[start:fade_start] += np.linspace(0, P[p, n], fade_start - start)
                    weights[end:fade_end] += np.linspace(P[p, n], 0, fade_end - end)

            pitch_son += weights * sin_tone

    pitch_son = pitch_son / np.max(np.abs(pitch_son))
    return pitch_son


def sonify_pitch_activations_with_signal(P,
                                         x,
                                         frame_rate,
                                         Fs,
                                         min_pitch=1,
                                         Fc=440,
                                         harmonics_weights=(1,),
                                         fading_msec=5,
                                         stereo=True):
    """Sonify a pitch activation matrix and combine it with a signal."""
    N = x.size
    pitch_son = sonify_pitch_activations(P, N, frame_rate, Fs, min_pitch=min_pitch, Fc=Fc,
                                         harmonics_weights=harmonics_weights, fading_msec=fading_msec)
    pitch_scaled = pitch_son * np.sqrt(np.mean(x ** 2)) / np.sqrt(np.mean(pitch_son ** 2))

    if stereo:
        out = np.vstack((x, pitch_scaled))
    else:
        out = x + pitch_scaled

    return pitch_son, out
