#
# audio.py
#
# Simple audio db for working with audio files and features
#

from __future__ import print_function
from __future__ import division
import extract
import glob
import os
import traceback
import re
import numpy as np
import pandas as pd
import librosa
from sklearn.preprocessing import MinMaxScaler

# labels for SVM classification
LABEL_MUSIC = 'yes'
LABEL_SPEECH = 'no'

# raw feature attribs with normalized bin counts from segmented metrics
ENERGY = '_df_energy'
ZCR = '_df_zcr'
SC = '_df_sc'
CHROMA = '_df_chroma'

# combined features, gaussian normalized and minmax scaled to 0..1
FEATURE = '_df_feature'

ALL_FEATURE_GROUPS = [ENERGY, ZCR, SC, CHROMA]

# file : prop
FILE_ENERGY = ['energy.csv', ENERGY]
FILE_ZCR = ['zcr.csv', ZCR]
FILE_SC = ['sc.csv', SC]
FILE_CHROMA = ['chroma.csv', CHROMA]
FILE_FEATURE = ['features.csv', FEATURE]

# feature files
FILES = [
    FILE_ENERGY,
    FILE_ZCR,
    FILE_SC,
    FILE_CHROMA,
    FILE_FEATURE,
]


class AudioFile:
    """
        id:             id of the entry for use in get_wave(id)
        name:           display name
        filepath:       filepath the wave was loaded from
        label:          label identifies the audio as music or speech
        ext:            dict for extended attributes
    """
    def __init__(self, id, name, filepath, label):
        self.id = id
        self.name = name
        self.filepath = filepath
        self.label = label
        self.ext = {}
        self._data = None

    def data(self):
        return self._data

    def load_data(self):
        try:
            data, sr = librosa.load(self.filepath, sr=None)
            self._data = AudioData(data, sr)
        except:
            self._data = None

        return self._data

    def free_data(self):
        self._data = None

class AudioData:
    """
        sr:    sample rate
        xtime: time domain x-axis (time in seconds)
        ytime: time domain y-axis (amplitude scaled to -1, 1 by max observed, not max bit-depth)
        xfreq: frequency domain x-axis (frequency in Hz)
        yfreq: frequency domain y-axis (amplitude)
    """
    def __init__(self, data, sr):
        xtime = np.arange(len(data)) / sr
        ytime = 1.0 * data / max(data.max(), abs(data.min()))
        xfreq = np.fft.rfftfreq(len(ytime), 1 / sr)
        yfreq = np.fft.rfft(ytime)

        self.sr = sr
        self.xtime = xtime
        self.ytime = ytime
        self.xfreq = xfreq
        self.yfreq = yfreq


class AudioDB:
    """
    Simple audio db for audio feature extraction

    Usage
    ---------
    db = AudioDB()
    db.load_waves()
    db.get_waves()
    db.get_wave()

    db.extract_features()
    db.save_features()
    db.load_features()
    db.get_feature_vectors()
    db.get_feature_ids()
    """
    def __init__(self):
        # AudioFiles indexed by id
        self._files = {}

        # audio feature data frames
        self._df_energy = None
        self._df_zcr = None
        self._df_sc = None
        self._df_chroma = None
        self._df_feature = None

    def load_waves(self, pathname):
        """
        Loads the audio files at pathname, expecting the filenames
        to be prefixed with the label 'mu' or 'sp' for SVM training

        :param pathname: path to load waves, e.g. 'audio/*.wav'
        """

        # alpha-num sort
        convert = lambda t: int(t) if t.isdigit() else t
        sort = lambda filepath: [convert(t) for t in re.split('([0-9]+)', filepath)]

        # populate from audio files in pathname
        filepaths = glob.glob(pathname)
        name_suffix = ''

        if len(filepaths) == 0:
            name_suffix = ' (missing)'
            # use features.csv as a backup if no audio files present
            filename = FILE_FEATURE[0]
            filepath = os.path.join("features", filename)
            if not os.path.exists(filepath):
                log('load waves: %s not found.' % filepath)
                return

            try:
                df = pd.read_csv(filepath, index_col=0)
                filepaths = df.index.values
                debug('loaded file info from %s' % filepath)
            except:
                log('Error loading %s:' % (filepath))
                log(traceback.format_exc())
                return

        # sorted filepaths for loading in order
        filepaths = sorted(filepaths, key=sort, reverse=False)

        for filepath in filepaths:
            # populate list of AudioFiles
            try:
                filename = os.path.split(filepath)[1]
                label = LABEL_MUSIC if filename.startswith("mu") else LABEL_SPEECH
                id = filename
                name = filename + name_suffix

                # add wave entry to db
                self._files[id] = AudioFile(id, name, filepath, label)
                debug('loaded %s' % filepath)
            except:
                log('Error processing file: %s' % filepath)
                log(traceback.format_exc())


    def get_waves(self, **kwargs):
        """
        Returns the list of audio files previously loaded

        :kwarg sortKey: custom key func passed to sorted() for custom sorting
        :kwarg sortReverse: reverse passed to sorted() for custom sorting
        :return: a new sorted list of AudioEntry
        """

        # get unsorted wave entries
        waves = self._files.values()

        # default sort func
        convert = lambda t: int(t) if t.isdigit() else t
        sort_by_name = lambda wav: [convert(t) for t in re.split('([0-9]+)', wav.name)]

        # kwargs with defaults
        sort_reverse = kwargs.get('sortReverse', False)
        sort_key = kwargs.get('sortKey', sort_by_name)

        return sorted(waves, key=sort_key, reverse=sort_reverse)
        return waves

    def get_wave(self, id):
        """
        Retrieves a wave by filename

        :param id: the id of the wave to get (e.g. from AudioEntry.id)
        :return: the AudioEntry with the specified id, or None if one doesn't exist.
        """
        return self._files.get(id, None)

    def extract_features(self, **kwargs):
        """
        Calculates the feature counts from the raw waves
        :param status: status callback
        :return: waves in the database have basic params applied in memory
        """

        # optional callback function to report status back to the caller since this is a long running operation
        status=kwargs.get('status', lambda msg: debug(msg))

        index = [item.id for item in self.get_waves()]

        #sampling window for segmentation
        fft_window = 1544
        hop_length = 386

        self._df_energy, extract_energy = extract.make_energy(index, fft_window, hop_length)
        self._df_zcr, extract_zcr = extract.make_zcr(index, fft_window, hop_length)
        self._df_sc, extract_sc = extract.make_sc(index, fft_window, hop_length)
        self._df_chroma, extract_chroma = extract.make_chroma(index)

        items = self.get_waves()
        if len(items) == 0:
            raise ValueException("No wav files to extract.")

        for item in items:
            status("Processing %s" % item.filepath)
            item.load_data()
            try:
                extract_energy(item)
                extract_zcr(item)
                extract_sc(item)
                extract_chroma(item)
            finally:
                item.free_data()

        self.calc_features_derived()

        status("Done.")

    def calc_features_derived(self):
        """
        Calculates complete feature vector and caches it in the database
        :return: True if calculated, otherwise False
        """

        items = self.get_waves()
        if len(items) == 0:
            raise ValueError('No wav files found.')

        index = [item.id for item in items]

        df = pd.DataFrame(index=index, dtype=float) #preallocate memory
        dfs = [df]
        for feature in ALL_FEATURE_GROUPS:
            dfNext = getattr(self, feature)
            if dfNext is None:
                raise ValueError('Feature %s not found.' % feature)

            dfs.append(dfNext)

        df = pd.concat(dfs, axis=1)
        debug(df)
        if df.empty:
            raise ValueError('No feature data found.')

        df = self.gaussian_normalize(df)

        #minmax scale for sklearn SVM
        scaler = MinMaxScaler()
        cols = df.columns.values
        df[cols] = scaler.fit_transform(df[cols])

        self._df_feature = df

        if df is None:
            return False
        else:
           return True

    def get_feature_vectors(self, **kwargs):

        if self._df_feature is None:
            raise ValueError('No feature data. Try extracting features.')

        feature_groups = kwargs.get('feature_groups', ALL_FEATURE_GROUPS)
        features = kwargs.get('features', None)

        if features is None:
            features = self.get_feature_ids(feature_groups=feature_groups)

        if len(features) == 0:
            raise ValueError('No features specified. Select one or more features.')

        df = self._df_feature[features]
        labels = np.array(map(lambda f: self.get_wave(f).label, df.index.values))

        return df, labels

    def get_feature_ids(self, **kwargs):

        # feature group to get features for
        feature_groups = kwargs.get('feature_groups', ALL_FEATURE_GROUPS)

        if len(feature_groups) == 0:
            raise ValueError('No feature groups selected. Check one or more feature groups.')

        features = []

        for feature in feature_groups:
            df = getattr(self, feature)
            if df is None:
                raise ValueError('Feature %s not found. Try extracting features.' % feature)

            features.extend(df.columns.values)

        return features

    def save_features(self, dir):
        """
        Saves the wave features from the in-memory db to csv files in the dir specified.

        :param dir: the directory name to save the feature csv files to
        """

        # create the output directory if one doesn't exist
        if not os.path.exists(dir):
            os.mkdir(dir)

        for f in FILES:
            filename, feature = f
            filepath = os.path.join(dir, filename)
            debug('Saving %s with %s...' % (filepath, feature))
            df = getattr(self, feature)

            if df is not None:
                df.to_csv(filepath)

    def load_features(self, dir):
        """
        Loads the features from flatfiles into the pathname

        :param dir: the directory name to load the files from
        :return: True if wave features found for all waves, otherwise False
        """

        for f in FILES:
            filename, feature = f
            filepath = os.path.join(dir, filename)
            setattr(self, feature, None)
            if not os.path.exists(filepath):
                log('load features: %s not found.' % (filepath))
                ok = False
                continue
            try:
                df = pd.read_csv(filepath, index_col=0)
                setattr(self, feature, df)
                debug('loaded %s for: %s' % (filepath, feature))
            except:
                log('Error loading %s:' % (filepath))
                log(traceback.format_exc())
                return False

        df = getattr(self, FILE_FEATURE[1])
        if df is None:
           self.calc_features_derived()

        # Check all values populated
        ok = True
        for f in FILES:
            df = None
            filename, feature = f
            df = getattr(self, feature)
            if df is None:
                debug('%s not found.' % filename)
                continue
            for item in self.get_waves():
                hist = df.ix[item.id]
                if hist is None:
                    log("MISSING %s for %s" % (f[1], item.id))
                    ok = False
                else:
                    pass
                    debug("FOUND %s for %s" % (f[1], item.id))

        return ok

    def gaussian_normalize(self, df):
        """
        applies gaussian normalization (standard score) for each wave feature (column)
        :param df:
        :return: new DataFrame with normalized values
        """

        df = df.copy()
        std = df.std()
        avg = df.mean()

        def make_normalize(i):
            """
            normalizer closure factory over std, avg, i
            :param i: the column feature index being processed
            :return: a normalizer function that can be used with pandas Series.map()
            """
            def normalize(x):
                """
                returns the gaussian normalized value for x
                :param x: the value to normalize
                :return: the normalized value
                """
                if (x - avg[i]) == 0:
                    return 0
                elif std[i] == 0:
                    raise ValueError("Unexpected stdev 0 for col %d" % i)
                else:
                    return (x - avg[i]) / std[i]

            return normalize

        # get the index feature column names
        cols = df.columns.values

        # process each feature column, one at a time
        for i in range(0, len(cols)):
            debug("Gaussian Normalize: %s" % cols[i])
            df[cols[i]] = df[cols[i]].map(make_normalize(i))

        return df


def logerr(e):
    log('Error: %s' % e)
    log(traceback.format_exc())
    return e


def log(msg):
    """
    Logs a message to the console

    :param msg: message to log
    """

    print(msg)


def debug(*msg):
    """
    Logs a debug message to the console

    :param msg: message to log
    """
    pass
    '''
    if len(msg) == 1:
        print(msg[0])
    else:
        print(msg)
    '''

