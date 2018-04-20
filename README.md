# tonality
A simple Python audio classifier to predict speech or music using sklearn, pandas, and librosa.

Audio features are extracted using segmentation and a sliding FFT window that generates metrics for narrow frames of audio. Summary statistics are calculated for the frames, and the frames are binned into histograms for each feature category.

The combined feature stats create a 51-dimension audio feature vector, that is Gaussian normalized and min/max scaled for use by sklearn, and exported by pandas to features.csv.

The auto-select feature uses sklearn’s recursive feature elimination (RFE) algorithm, with cross validation (CV), to select an optimal set of features and rank them. Auto-selected features can be manually fine-tuned by selecting or deselecting features in the program. You can manually tune the audio features used in the GUI.

Cross-validated SVM test results are created with sklearn’s SVM package to evaluate feature selection.

Make builds a self-contained MacOS app and requires python 2.7, pip, and virtualenv.
