#
# App.py
#
# Simple app to classify audio wav files as speech or audio
# using librosa audio feature extraction and sklearn SVM
#
# 1. extract audio features and save intermediate results to csv files, using segmented windows sliding
# 2. collect stats (e.g. mean, stdev, var) and bin the data using bins with a resolution useful for analysis
# 3. create a 51-dimension feature vector (audio representation), gaussian normalized, and min/max scaled
# 4. auto-select the best features using RFECV (recursive feature elimnination with cross validation)
# 5. train a support vector machine to classify the waves as speech or music and test the results

from audio import AudioDB, ENERGY, ZCR, SC, CHROMA, log, logerr, debug
import tkMessageBox
import tkFileDialog
from Tkinter import *
from operator import itemgetter
import classify
import numpy as np
import time
import os
import sys
import subprocess
import matplotlib
if sys.platform == 'darwin':
    # work around darwin matplotlib issues
    import FileDialog
    matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import librosa
import librosa.display


class App(Frame):
    """
    root:       Tk() root
    db:         AudioDB
    resultWin:  top level Tk() window to display results in
    waves:      list of waves sorted by filepath
    """
    def __init__(self, root, db):
        Frame.__init__(self, root)
        self.root = root
        self.db = db
        self.resultWin = None
        self.resultsText = None

        self.waves = db.get_waves()

        # Audio listbox
        Label(root,
            text="Available Wave Files:",
            justify=LEFT).grid(
            row=0,
            sticky=NW)

        lframe = Frame(root)
        lframe.grid(
            row=1,
            padx=5,
            sticky=W)

        self.listScrollbar = Scrollbar(lframe)
        self.listScrollbar.pack(
            side=RIGHT,
            fill=Y)

        self.list = Listbox(lframe,
            yscrollcommand=self.listScrollbar.set,
            selectmode=BROWSE,
            height=14)

        Button(root,
               text="Select Directory",
               command=lambda: self.select_directory(),
               width=12).grid(
            row=2,
            padx=0,
            sticky=W)

        self.list.configure(exportselection=False)


        self.list.pack(
            side=RIGHT,
            fill=Y)

        self.listScrollbar.config(command=self.list.yview)

        # Status bar
        self.statusMsg = StringVar()
        Label(root,
            textvariable=self.statusMsg,
            justify=LEFT).grid(
            row=3,
            columnspan=6,
            sticky=SW)

        # Layout buttons
        button_pad = 10
        button_width = 20

        bframe = Frame(root)
        bframe.grid(
            row=1,
            column=2,
            rowspan=2,
            padx=5)

        Button(bframe,
               text="Listen",
               command=self.listen_sel_wave,
               width=button_width).grid(
            row=0,
            padx=button_pad,
            sticky=E)

        Button(bframe,
               text="Plot Time Domain",
               command=self.time_sel_wave,
               width=button_width).grid(
            row=1,
            padx=button_pad,
            sticky=E)

        Button(bframe,
               text="Plot Frequency Domain",
               command=self.freq_sel_wave,
               width=button_width).grid(
            row=2,
            padx=button_pad,
            sticky=E)

        Button(bframe,
               text="Plot Chromagram",
               command=self.chroma_sel_wave,
               width=button_width).grid(
            row=3,
            padx=button_pad,
            sticky=E)

        Button(bframe,
              text="Extract Features",
              command=lambda: self.extract_features(),
              width=button_width).grid(
            row=4,
            padx=button_pad,
            sticky=E)

        Button(bframe,
               text="Auto-Select Features",
               command=lambda: self.auto_select_features(),
               width=button_width).grid(
            row=5,
            padx=button_pad,
            sticky=E)


        Label(bframe,
              text="Feature Groups:",
              pady=0,
              anchor=N,
              justify=LEFT).grid(
            row=6,
            sticky=W)

        # audio feature groups that contain multiple features each
        self.use_energy = IntVar(value=1)
        Checkbutton(bframe,
                    text="Energy",
                    command=lambda: self.update_features_list(),
                    justify=LEFT,
                    pady=0,
                    anchor=N,
                    variable=self.use_energy,
                    fg="#777777").grid(
            row=7,
            padx=button_pad,
            sticky=W)

        self.use_zcr = IntVar(value=1)
        Checkbutton(bframe,
                    text="Zero Crossing Rate",
                    command=lambda: self.update_features_list(),
                    justify=LEFT,
                    pady=0,
                    anchor=N,
                    variable=self.use_zcr,
                    fg="#777777").grid(
            row=8,
            padx=button_pad,
            sticky=W)

        self.use_sc = IntVar(value=1)
        Checkbutton(bframe,
                    text="Spectral Centroid",
                    command=lambda: self.update_features_list(),
                    justify=LEFT,
                    pady=0,
                    anchor=N,
                    variable=self.use_sc,
                    fg="#777777").grid(
            row=9,
            padx=button_pad,
            sticky=W)

        self.use_chroma = IntVar()
        Checkbutton(bframe,
                    text="Chroma",
                    command=lambda: self.update_features_list(),
                    justify=LEFT,
                    pady=0,
                    anchor=N,
                    variable=self.use_chroma,
                    fg="#777777").grid(
            row=10,
            padx=button_pad,
            sticky=W)

        # Features listbox
        Label(root,
              text="Select Audio Features:",
              justify=LEFT).grid(
            row=0,
            column=4,
            sticky=NW)

        #audio features list box
        lframe2 = Frame(root)
        lframe2.grid(
            row=1,
            column=4,
            padx=5,
            sticky=W)

        self.listScrollbar2= Scrollbar(lframe2)
        self.listScrollbar2.pack(
            side=RIGHT,
            fill=Y)

        self.list2 = Listbox(lframe2,
                            yscrollcommand=self.listScrollbar2.set,
                            selectmode=MULTIPLE,
                            height=14)

        self.list2.configure(exportselection=False)

        #actions on the selected audio features
        Button(root,
               text="Test Results",
               command=lambda: self.show_test_results(),
               width=10).grid(
            row=2,
            column=4,
            padx=0,
            sticky=E)

        Button(root,
               text="Clear",
               command=lambda: self.select_set_feature_ids(),
               width=6).grid(
            row=2,
            column=4,
            padx=0,
            sticky=W)

        self.list2.pack(
            side=RIGHT,
            fill=Y)

        self.listScrollbar2.config(command=self.list2.yview)

    def get_sel_feature_groups(self):
        """
        Get a list of the selected feature groups
        :return: list of selected feature groups
        """
        fg = []

        if self.use_energy.get():
            fg.append(ENERGY)

        if self.use_zcr.get():
            fg.append(ZCR)

        if self.use_sc.get():
            fg.append(SC)

        if self.use_chroma.get():
            fg.append(CHROMA)

        return fg

    def get_sel_features(self):
        """
        Get a list of the selected features in the audio features listbox
        :return: list of selected features
        """
        idx = self.list2.curselection()
        if len(idx) == 0:
            return []

        if len(idx) == 1:
            return [self.feature_ids[idx[0]]]

        return list(itemgetter(*idx)(self.feature_ids))

    def select_set_feature_ids(self, selected=None):
        """
        Sets the selected items in the features listbox
        :param selected: the features to select
        :return: n/a
        """
        self.list2.select_clear(0, END)

        if selected is None:
            return

        for sel in selected:
            try:
                 idx = self.feature_ids.index(sel)
                 self.list2.select_set(idx)
            except Exception as e:
                logerr(e)
                continue

        self.root.update()

    def select_directory(self, selected=None):
        """
        Sets the selected items in the features listbox
        :param selected: the features to select
        :return: n/a
        """
        dirpath = tkFileDialog.askdirectory()
        if dirpath:
            debug('Selected dirpath %s' % dirpath)
            self.load_waves(dirpath)

    def update_features_list(self, **kwargs):
        """
        Updates the feature list with a new set of features
        :kwarg features: the feature ids
        :kwarg features_labels: the optional associated feature_labels, if None, feature ids will be used as labels
        :return: n/a
        """
        features=kwargs.get('features', None)
        feature_labels=kwargs.get('feature_labels', None)

        try:
            fg = self.get_sel_feature_groups()

            if features is None:
                features = self.db.get_feature_ids(feature_groups=fg)

            if feature_labels is None:
                feature_labels = features

            #linked by index
            self.feature_ids = features
            self.feature_labels = feature_labels

            self.list2.delete(0, END)
            for f in feature_labels:
                self.list2.insert(END, f)

        except Exception as e:
            logerr(e)
            self.status_update('Update features: %s' % e)

        self.root.update()


    def update_status(self, msg):
        """
        Updates the status bar at the bottom of the control window

        :param msg: the message to display in the status bar
        """
        maxlen = 75
        sidelen = 36
        if len(msg) > maxlen:
            msg = '%s...%s' % (msg[:sidelen], msg[len(msg)-sidelen:])

        self.statusMsg.set('Status: ' + msg)
        self.root.update()

    def selected_wav(self):
        """
        Gets the selected wav, or None.
        :return: the selected wav
        """
        wav = None
        if len(self.waves) > 0:
            i = self.list.curselection()[0]
            wav = self.waves[int(i)]

        return wav

    def listen_sel_wave(self):
        """
        Opens the selected picture with default os viewer
        """
        wav = self.selected_wav()
        if wav is None:
            tkMessageBox.showwarning(
                'Listen',
                'No wav file selected.')
            return

        filepath = wav.filepath
        if filepath is None:
            return

        if not os.path.exists(filepath):
                tkMessageBox.showwarning(
                    "Listen",
                    '%s not found.' % filepath)
                return

        # handle different platforms
        if sys.platform == "darwin":
            subprocess.call(["open", filepath])
        elif sys.platform == "win32":
            os.startfile(filepath)
        else:
            subprocess.call(["xdg-open", filepath])

    def time_sel_wave(self):
        """
        Display matplotlib graph of time domain chart for the selected wav file
        :return: n/a
        """
        wav = self.selected_wav()
        if wav is None:
            tkMessageBox.showwarning(
                'Plot Time Domain',
                'No wav file selected.')
            return

        wav.load_data()
        try:
            data = wav.data()
            if data is None:
                tkMessageBox.showwarning(
                    "Plot Time Domain",
                    '%s not found.' % wav.filepath)
                return

            fig = plt.figure()
            sp = fig.add_subplot(111)
            sp.plot(data.xtime, data.ytime)
            plt.xlabel('Time')
            plt.ylabel('Amplitude')
            plt.title('Time Domain: %s' % wav.name)
            plt.show()

        finally:
            wav.free_data()

    def freq_sel_wave(self):
        """
        Display matplotlib graph of frequency domain chart for the selected wav file
        :return: n/a
        """
        wav = self.selected_wav()
        if wav is None:
            tkMessageBox.showwarning(
                'Plot Frequency Domain',
                'No wav file selected.')
            return

        wav.load_data()
        try:
            data = wav.data()
            if data is None:
                tkMessageBox.showwarning(
                    'Plot Frequency Domain',
                    '%s not found.' % wav.filepath)
                return

            fig = plt.figure()
            sp = fig.add_subplot(111)
            sp.plot(data.xfreq, np.abs(data.yfreq))
            plt.xlabel('Freq')
            plt.ylabel('Amplitude')
            plt.title('Frequency Domain: %s' % wav.name)
            plt.show()

        finally:
            wav.free_data()

    def chroma_sel_wave(self):
        """
        Display librosa chroma graphs of time domain chart for the selected wav file
        :return: n/a
        """
        wav = self.selected_wav()
        if wav is None:
            tkMessageBox.showwarning(
                'Plot Chromagram',
                'No wav file selected.')
            return

        wav.load_data()
        try:
            data = wav.data()
            if data is None:
                tkMessageBox.showwarning(
                    'Plot Chromagram',
                    '%s not found.' % wav.filepath)
                return

            #chroma stft (short time fourier xform)
            chroma_stft = librosa.feature.chroma_stft(y=data.ytime, sr=data.sr, n_chroma = 12, n_fft = 4096)
            plt.figure()
            plt.subplot(2, 1, 1)
            librosa.display.specshow(chroma_stft, y_axis='chroma')
            plt.title('chroma_stft: %s' % wav.name)
            plt.xlabel('Time')
            plt.ylabel('Frequency')
            plt.colorbar()

            #chroma cqt (constant-Q xform) for comparison
            chroma_cq = librosa.feature.chroma_cqt(y=data.ytime, sr=data.sr)
            plt.subplot(2, 1, 2)
            librosa.display.specshow(chroma_cq, y_axis='chroma', x_axis='time')
            plt.title('chroma_cqt: %s' % wav.name)
            plt.xlabel('Time')
            plt.ylabel('Frequency')
            plt.colorbar()
            plt.tight_layout()
            plt.show()

        finally:
            wav.free_data()

    def extract_features(self):
        """
        Process_waves calculates the wav feature vectors,
        stores them in the wav database and
        saves them to disk as csv files
        """

        t = Timer()
        self.update_status('Extracting features ...')
        try:
            self.db.extract_features(status=lambda msg: self.update_status(msg))
            self.db.save_features('features')
            self.update_status('Processed waves and saved features to disk in %d ms.' % (t.mark()))
            self.update_features_list()
        except Exception as e:
            logerr(e)
            self.update_status("Feature extraction failed for some files. Try a different source directory.")
            tkMessageBox.showwarning(
                'Extract Features',
                '%s' % e)
            return

    def auto_select_features(self):
        """
        Uses sklearn to auto select the best features for the selected feature groups
        using recursive feature elimination and cross validation. The best features
        are selected in the audio features list and the list is updating with a feature ranking score
        next to each feature, to aid in manual tuning of the auto-selected features.
        A matplotlib chart is displayed that shows the auto selection grid search result vs. accuracy.
        :return:
        """

        t = Timer()
        self.update_status('Auto-selecting features ...')

        # calculate the optimal features and get features and labels (with ranking scores)
        try:
            fg = self.get_sel_feature_groups()

            if len(fg) == 0:
                tkMessageBox.showwarning(
                    'Auto-Select Features',
                    'Select one or more feature groups, then click Auto-Select Features.')
                self.update_status('Ready.')
                return

            df, labels = self.db.get_feature_vectors(feature_groups=fg)
            if df is None or df.empty:
                tkMessageBox.showwarning(
                    'Auto-Select Features',
                    'No features data found, try extracting features again.')
                self.update_status('Ready.')
                return

            best_features, features, feature_ranking, grid_scores = classify.auto_select_features(df, labels)
            if best_features is None:
                tkMessageBox.showwarning(
                    'Auto-Select Features',
                    'No best features found, try extracting features again.')
                self.update_status('Ready.')
                return

            feature_labels = ['%s (%d)' % (features[i], feature_ranking[i]) for i in range(0, len(features), 1)]

            # update audio features listbox with auto selected features and updated ranking labels
            self.update_features_list(features=features, feature_labels=feature_labels)
            self.select_set_feature_ids(best_features)

            self.update_status('Features auto-selected in %d ms.' % (t.mark()))

            # show auto select grid search results
            plt.figure()
            plt.xlabel("Number of features selected")
            plt.ylabel("Cross validation score (nb of correct classifications)")
            plt.plot(range(1, len(grid_scores) + 1), grid_scores)
            plt.title('Auto-Select Features results.')
            plt.show()
        except Exception as e:
            logerr(e)
            self.update_status('Auto-select error: %s' % e)
            tkMessageBox.showwarning(
                'Auto-Select Features',
                '%s' % e)

    def show_test_results(self):
        """
        Computes cross validated test results and displays the results
        in a results windows.
        :return:
        """
        features = self.get_sel_features()
        if len(features) == 0:
            self.update_status('No features specified. Select one or more audio features.')
            return

        def status(*msg):
            for m in msg:
                self.resultsText.insert(END, '%s\n' % m)

            self.resultsText.see(END)

        if self.resultsText is not None:
            self.resultsText.delete(1.0, END)

        t = Timer()
        self.update_status('Calculating test results ...')

        try:
            self.show_results_window()

            df, labels = self.db.get_feature_vectors(features=features)

            classify.train_svc(df, labels, status=status)

            self.update_status('Test results generated in %d ms.' % (t.mark()))
        except Exception as e:
            logerr(e)
            e = logerr(sys.exc_info()[0])
            self.update_status('Test error: %s' % e)


    def close_result_win_handler(self):
        """
        Cleans up when the results window is closed.
        """
        if self.resultWin is None:
            return

        self.resultWin.destroy()
        self.resultWin = None
        self.resultsText = None

    def show_results_window(self):
        """
        Show the results window with the results text.
        """

        if self.resultWin is None:
            # Results window
            resultWin = Toplevel(self.root)
            resultWin.title('Result Viewer')

            resultWin.protocol('WM_DELETE_WINDOW', lambda: self.close_result_win_handler())
            resultWin.geometry('+%d+%d' % (self.root.winfo_x(), self.root.winfo_y() + 325))

            t = Text(resultWin,
                height=30,
                width=100,
                bd=0)

            s = Scrollbar(resultWin)
            s.pack(side=RIGHT, fill=Y)
            t.pack(expand=True, fill=BOTH)
            s.config(command=t.yview)
            t.config(yscrollcommand=s.set)

            self.resultsText = t
            self.resultWin = resultWin

        self.resultWin.lift()

    def load_features(self):
        # Load cached wav feature csv files if they exist
        try:
            items = self.db.get_waves()
            if len(items) > 0:
                self.update_status('Loading audio features ...')
                ok = self.db.load_features('features')
                if ok:
                    self.update_status('%d waves loaded. Found cached waves features for all waves.' % len(items))
                else:
                    self.update_status('%d waves loaded. Some audio features missing. Click Extract Features.' % len(items))

                self.update_features_list()

            else:
                self.update_status('%d waves loaded. Select a different directory and extract features.' % len(items))
        except Exception as e:
            logerr(e)
            self.update_status('Load error: %s' % e)

    def load_waves(self, dirpath):
        self.db.load_waves('%s/*.wav' % dirpath)
        self.waves = self.db.get_waves()

        # Populate available waves listbox
        self.list.delete(0, END)
        for wav in self.waves:
            self.list.insert(END, wav.name)

        # Make first wav list selected and active
        if len(self.waves) > 0:
            self.list.activate(0)
            self.list.select_set(0)
            self.list.event_generate("<<ListboxSelect>>")

        self.load_features()

        self.root.update()

    def run(self):
        self.load_waves('audio')

        self.root.mainloop()


class Timer:
    """
    Timer measures time elapsed in milliseconds used to display the
    time it takes the app to process waves or retrieve results.
    """
    def __init__(self):
        self.stamp = self._current()

    def _current(self):
        """
        Gets time in milliseconds
        :return: current time in milliseconds
        """
        return int(round(time.time() * 1000))

    def mark(self):
        """
        Gets the elapsed time since Timer() was created or the last mark() call
        :return: time elapsed in milliseconds
        """
        start = self.stamp
        end = self._current()
        diff = end - start
        self.stamp = end
        return diff


def start():

    # Initialize GUI
    root = Tk()

    if sys.platform == 'darwin':
        menubar = Menu(root)
        appmenu = Menu(menubar, name='apple')
        menubar.add_cascade(menu=appmenu)
        root['menu'] = menubar

    root.title('Audio Classifier')

    # Initialize database
    db = AudioDB()
    gui = App(root, db)
    gui.run()


if __name__ == '__main__':
   start()
