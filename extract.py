#
# extract.py
#
# Feature extractors for energy, zero crossing, spectral centroid, and chroma
#

from __future__ import print_function
from __future__ import division
import math
import numpy as np
import pandas as pd
import scipy.stats as stats
import librosa

# energy
def make_energy(index, fft_window, hop_length):

    E_STD = 'e:std'
    E_VAR = 'e:var'
    E_MAX = 'e:max'
    E_MEAN = 'e:mean'

    bins = np.array([0., 4., 8., 16., 32., 64., 128., 256., 512., 2048.])

    cols = np.concatenate( (
        ['e:%.0f-%.0f' % (bins[i], bins[i+1]) for i in range(len(bins) - 1)],
        [E_STD, E_VAR, E_MAX, E_MEAN]))

    # preallocate memory
    df = pd.DataFrame(index=index, columns=cols, dtype=float)
    df.index.name = 'Energy'

    def extract(item):
        if item.data() is None:
            raise ValueError('Unable to load data for %s.' % item.filepath)

        raw = np.array([
            sum(abs(item.data().ytime[i:i+fft_window]**2))
            for i in range(0, len(item.data().ytime), hop_length)
        ])
        s = [stats.tstd(raw), stats.tvar(raw), stats.tmax(raw), stats.tmean(raw)]
        # energy bins tested that show the distribution of energy over all the frames
        frames = len(raw)
        hist = np.histogram(raw, bins)[0] / frames #normalized
        values = np.concatenate([hist, s])
        if len(values) != len(cols):
            raise ValueError('Feature length mismatch, try extracting features again.')
        df.ix[item.id] = values

    return df, extract


# zero crossing rate
def make_zcr(index, fft_window, hop_length):

    Z_BIMODAL = 'z:bimodal'
    Z_STD = 'z:std'
    Z_VAR = 'z:var'

    bins = np.array([0, .065, 0.090, 0.100, 0.115, 0.130, 0.140, 0.165, 100])
    cols = np.concatenate( (
        ['z:%.3f-%.3f' % (bins[i], bins[i+1]) for i in range(len(bins) - 1)],
        [Z_BIMODAL, Z_STD, Z_VAR]))

    #preallocate memory
    df = pd.DataFrame(index=index, columns=cols, dtype=float)
    df.index.name = 'ZCR'

    def extract(item):
        if item.data() is None:
            raise ValueError('Unable to load data for %s.' % item.filepath)

        raw = librosa.feature.zero_crossing_rate(item.data().ytime + 0.0001, fft_window, hop_length)[0]
        frames = len(raw)
        s = [stats.tstd(raw), stats.tvar(raw)]

        # zero crossing rate bins tested that show the distribution of zcr over all the frames
        hist = np.histogram(raw, bins=bins)[0] / frames #normalized to pct of total
        bimodal_edge_score = score_bimodal_edge_peak_histogram(hist)
        values = np.concatenate([hist, [bimodal_edge_score], s])
        if len(values) != len(cols):
            raise ValueError('Feature length mismatch, try extracting features again.')
        df.ix[item.id] = values

    return df, extract


# spectral centroid
def make_sc(index, fft_window, hop_length):

    SC_BIMODAL = 'sc:bimodal'
    SC_STD = 'sc:std'
    SC_MEAN = 'sc:mean'

    bins = np.array([0, 1200, 1400, 1600, 1800, 2000, 3000, 3500, 8000])

    cols = np.concatenate( (
        ['sc:%.0f-%.0f' % (bins[i], bins[i+1]) for i in range(len(bins) - 1)],
        [SC_BIMODAL, SC_MEAN, SC_STD]))

    #preallocate memory
    df = pd.DataFrame(index=index, columns=cols, dtype=float)
    df.index.name = 'SC'

    def extract(item):
        if item.data() is None:
            raise ValueError('Unable to load data for %s.' % item.filepath)

        raw = librosa.feature.spectral_centroid(item.data().ytime, item.data().sr, None, fft_window, hop_length)[0]
        frames = len(raw)
        s = [stats.tstd(raw), stats.tmean(raw)]

        #spectral centroid frequency bins to see the distribution of center frequency for all frame
        hist = np.histogram(raw, bins=bins)[0] / frames #normalized to pct of total
        bimodal_edgeore = score_bimodal_edge_peak_histogram(hist) #detect bimodal edge peak distribution
        values = np.concatenate([hist, [bimodal_edgeore], s])
        if len(values) != len(cols):
            raise ValueError('Feature length mismatch, try extracting features again.')
        df.ix[item.id] = np.concatenate([hist, [bimodal_edgeore], s])

    return df, extract


# chroma
def make_chroma(index):

    C_TONETOP = 'c:tonemax'
    C_TONE2TOP = 'c:tonemax2'
    C_TONE3TOP = 'c:tonemax3'
    C_TONE4TOP = 'c:tonemax4'

    bins = np.arange(12) #chroma features are extracted into 12 bins
    cols = np.concatenate( (
        ['c:%d' % (i) for i in bins],
        [C_TONETOP, C_TONE2TOP, C_TONE3TOP, C_TONE4TOP] ))

    #preallocate memory
    df = pd.DataFrame(index=index, columns=cols, dtype=float)
    df.index.name = 'Chroma'

    def extract(item):
        if item.data() is None:
            raise ValueError('Unable to load data for %s.' % item.filepath)

        # tonality stats: percentage of frames spent in dominant tone
        # for music more time is spent in the dominant tone
        # for speech it's more scattered
        raw = librosa.feature.chroma_stft(item.data().ytime, sr=item.data().sr)
        peak_tone_counts = np.sum(raw == 1, axis=1)
        peak_tone_counts = peak_tone_counts / np.sum(peak_tone_counts) #normalized to pct of total
        dominant_tone_count = np.max(peak_tone_counts)
        sorted_tone_counts = sorted(peak_tone_counts, reverse=True)
        top2_tone_counts = sorted_tone_counts[0] + sorted_tone_counts[1]
        top3_tone_counts = top2_tone_counts + sorted_tone_counts[2]
        top4_tone_counts = top3_tone_counts + sorted_tone_counts[3]
        values = np.concatenate([ peak_tone_counts, [dominant_tone_count, top2_tone_counts, top3_tone_counts, top4_tone_counts] ])
        if len(values) != len(cols):
            raise ValueError('Feature length mismatch, try extracting features again.')

        df.ix[item.id] = values

    return df, extract


def score_bimodal_edge_peak_histogram(h):
    frames = np.sum(h)
    bin1 = h[0]/frames * 100
    binN = h[len(h)-1]/frames * 100
    if bin1 > 0:
        #de-emphasise larger values and boost moderate values, such that bin1*binN is larger for two more equal values
        #vs. a single large value and a smaller value
        bin1 = math.log10(1 + bin1**2)
    if binN > 0:
        #de-emphasise larger values and boost moderate values
        binN = math.log10(1 + binN**2)

    #return distribution score - maximize value for peaks at both sides of dist, not just one side
    return (bin1*binN+bin1+binN) / 100


