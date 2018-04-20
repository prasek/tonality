# tonality
A simple Python audio classifier to predict speech or music using sklearn, pandas, and librosa.

Audio features are extracted using segmentation and a sliding FFT window that generates metrics for narrow frames of audio. Summary statistics are calculated and the frames are binned into histograms for each feature category.

The combined feature stats create a 48-dimension audio feature vector, that is Gaussian normalized and min/max scaled for use by sklearn, and exported by pandas to features.csv.

The auto-select feature uses sklearn’s recursive feature elimination (RFE) algorithm, with cross validation (CV), to select an optimal set of features and rank them. Auto-selected features can be manually fine-tuned by selecting or deselecting features in the program. You can manually tune the audio features used in the GUI.

Cross-validated SVM test results are created with sklearn’s SVM package to evaluate feature selection.

A self-contained MacOS app can be built with make. You'll need python 2.7, pip, and virtualenv.

```sh
$ make
```
