# Classification of TTS Audio Data

## Problem Statement

Speech synthesis has become increasingly sophisticated, with computer-generated voices that can sound remarkably human-like. However, distinguishing between synthetic speech and human speech can still be challenging, especially in scenarios where the quality of the audio signal may be poor or the characteristics of the speakers are highly variable. In this project, we aim to develop a machine learning model to classify speech signals as either human or synthetic, based on their acoustic features.

To accomplish this, we will use a dataset of voice recordings that includes samples of both human and synthetic speech. We will preprocess the audio signals to extract a set of relevant features, such as spectral density, pitch, and formant frequencies. We will then train a machine learning model, such as a random forest classifier or a support vector machine (SVM), to classify speech signals as either human or synthetic, based on these features.

The performance of the model will be evaluated using metrics such as accuracy, precision, recall, and F1 score. We will also conduct a thorough analysis of the model's performance, including a confusion matrix, to determine which classes are being misclassified and whether any particular features are driving the classification decision.

Overall, the goal of this project is to develop a machine learning model that can accurately classify speech signals as either human or synthetic, and to gain insights into the underlying features that contribute to this classification. The resulting model could have applications in fields such as speech recognition, natural language processing, and voice authentication, among others.

--- 

## Datasets used

Four datasets were used for this analysis and their contents were compiled into a single dataset that was passed into our models.

The first dataset is called SOMOS, which was created by DARPA as part of its TRANSTAC program. It contains over 250,000 utterances, including 20,000 synthetic utterances in WAV file format and 100 natural utterances. The dataset also includes 374,955 human-assigned scores, which range from 1 to 5, to evaluate the naturalness of the utterances. SOMOS is widely used by researchers to develop and evaluate machine learning models for speech recognition and machine translation.

The second dataset is called the Mozilla CommonVoice dataset. It is a collection of over 9,000 hours of speech from more than 9,000 unique speakers across 60 languages. We'll be using a subset of the dataset, which includes approximately 1000 hours of speech. Each recording in the dataset is accompanied by metadata such as the speaker's age and gender, and additional information such as the recording location and the text of the spoken utterance. The CommonVoice dataset has been widely used in the research community to train and evaluate speech recognition models.

Thirdly, there's the Real or Fake dataset from APTLY labs, which is a collection of approximately 2,000 unique speech utterances labeled as either real or fake. The fake utterances were generated using a voice conversion algorithm that can transform the voice of one person to sound like another person. The dataset includes both male and female speakers, and the real utterances are drawn from a variety of sources including the LibriSpeech and Common Voice datasets. The Real or Fake dataset is intended for research purposes and can be used to train and evaluate machine learning models for detecting fake speech.

Lastly,The LibriTTS corpus is a collection of speech recordings and accompanying text transcripts, designed to support research in speech synthesis and related fields. The dataset consists of approximately 5,000 hours of speech, recorded by over 2,500 speakers. The speech recordings were taken from the LibriSpeech corpus, which contains audio recordings of public domain books read by volunteers. The text transcripts were aligned to the speech recordings using forced alignment techniques.

Each recording in the dataset is accompanied by metadata such as the speaker's gender, age, and accent, as well as information about the book and chapter from which the recording was taken. The dataset also includes precomputed features such as Mel-spectrograms and pitch contours, as well as a variety of other metadata such as language and accent labels.

A deeper explanation of these datasets is provided within '00. Project Introduction'.

--- 

## Data Dictionary

| Column Name   | Description                                                                                                                                                                      | Data Type   |
|---------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------|
| utteranceId   | Id of the utterance, composed from sentenceId and systemId.                                                                                                                      | string      |
| choice        | 1-5: choice of listener in a Likert-type scale from very unnatural (1) to completely natural (5).                                                                                | integer     |
| sentenceId    | Original Id of sentence text. Includes info on text domain.                                                                                                                      | string      |
| systemId      | 000-200: 000 is natural speech, 001-200 are TTS systems.                                                                                                                          | string      |
| modelId       | m0-m5: m0 is natural speech, m1-m5 are TTS models.                                                                                                                                | string      |
| testpageId    | 000-999: corresponds to HIT Id on Amazon Mechanical Turk.                                                                                                                        | string      |
| locale        | us (United States), gb (United Kingdom), ca (Canada): registered locale of listener on Amazon Mechanical Turk.                                                                   | string      |
| listenerId    | Anonymized AMT worker Id.                                                                                                                                                         | string      |
| isNative      | 0 (no), 1 (yes): Although only residents of the respective English locale, according to AMT’s qualification, were recruited in each test, and only native English speakers were asked to participate, a native/non-native annotation checkbox was included in each test page. | integer     |
| wrongValidation | 0 (fails to pass quality check), 1 (passes quality check): Wrong score has been assigned to the validation sample on test page. Validation utterances and respective choices have been removed from the dataset, but the page-level validation annotation has been propagated for every choice in the page. | integer |
| lowNatural    | 0 (fails to pass quality check), 1 (passes quality check): The score assigned to the natural sample on test page is extremely low (1 or 2).                                      | integer     |
| sameScores    | 0 (fails to pass quality check), 1 (passes quality check): All scores on test page are identical ignoring the score of the validation utterance.                                     | integer     |
| highSynthetic | 0 (fails to pass quality check), 1 (passes quality check): The average score of synthetic samples on page is higher or close (down to smaller by 0.1) to the natural sample's score.                                                               | integer     |
| clean         | 0 (no), 1 (yes): Clean is the logical AND of the 4 quality checks (wrongValidation, lowNatural, sameScores, highSynthetic) that have been used in the dataset per test page in Flag 0/1 form. Thus, all test pages that have passed all 4 quality checks are considered clean. | integer     |
| listenerReliability | The percentage of test pages that the listener has submitted and are considered clean, expressed in the range [0, 1].                                                             | float       |
| path          | Relative path to the audio file in the dataset                                                                                                                                   | string      |
| sentence_id   | ID of the corresponding sentence                                                                                                                                                 | integer     |
| up_votes      | Number of up votes for the recording                                                                                                                                              | integer     |
| down_votes    | Number of down votes for the recording                                                                                                                                            | integer     |
| age           | Age of the speaker who recorded the sentence                                                                                                                                      | integer     |
| gender        | Gender of the speaker who recorded the sentence                                                                                                                                   | string      |
| accent        | Accent of the speaker who recorded the sentence                                                                                                                                   |
| id            | Unique ID of the utterance in the dataset                  |
| text          | Text transcription of the spoken utterance                  |
| speaker_id    | Unique ID of the speaker who recorded the utterance         |
| chapter_id    | ID of the book chapter from which the utterance was derived |
| book_id       | ID of the book from which the utterance was derived         |
| dataset_id    | ID of the dataset from which the utterance was derived      |
| duration      | Duration of the audio file in seconds                       |
| audio_path    | File path to the audio file in WAV format                    |
| normalized    | Binary indicator of whether the audio has been normalized   |
| split         | Training, validation, or test split of the data              |

--- 
--- 

### Resources

1. https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html
2. https://librosa.org/doc/main/index.html
3. https://www.sciencedirect.com/topics/engineering/zero-crossing-rate
4. https://medium.com/@LeonFedden/comparative-audio-analysis-with-wavenet-mfccs-umap-t-sne-and-pca-cb8237bfce2f
5. https://machinelearning.apple.com/research/mel-spectrogram
6. https://www.kaggle.com/code/shivamburnwal/speech-emotion-recognition
7. https://medium.com/heuristics/audio-signal-feature-extraction-and-clustering-935319d2225
8. https://www.soundjay.com/ambient-sounds.html
7. https://www.openslr.org/60/
8. https://bil.eecs.yorku.ca/datasets/
9. https://github.com/jeffprosise/Deep-Learning/blob/master/Audio%20Classification%20(CNN).ipynb