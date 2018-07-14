# Overview

This repo is currently a work in progress to make Deep Learning on EEG more accessible!

# Datasets
There exist multiple available (public) EEG datasets.

Here are 2 good list of public datasets:
1. [Open Datasets in Human Electrophysiology](https://github.com/voytekresearch/OpenData) (Tom Donoghue, VoytekLab)
2. [EEG / ERP data available for free public download](https://sccn.ucsd.edu/~arno/fam2data/publicly_available_EEG_data.html) (UCSD)

This series of workshops will focus on 5 categories: Sleep, Epilepsy, BCI, Emotions, Workload

## Sleep
* [Sleep Heart Health Study (SHHS)](https://sleepdata.org/datasets/shhs)  (access required - requested)
* [MASS Dataset](https://massdb.herokuapp.com/en/)  (access required)
* [Sleep EDF](https://www.physionet.org/physiobank/database/sleep-edf/)
* [Sleep EDFx](https://www.physionet.org/pn4/sleep-edfx/)
* [Sleep EDFxx](https://www.physionet.org/pn6/sleep-edfxx/)
* [MIT-BIH Polysomnographic](https://www.physionet.org/physiobank/database/slpdb/)
* [More Sleep Datasets here](https://sleepdata.org/datasets)

## Epilepsy
* [Bonn University Dataset](http://epileptologie-bonn.de/cms/front_content.php?idcat=193&lang=3&changelang=3)
* [CHB-MIT Dataset](https://physionet.org/pn6/chbmit/)
* [Boston Children's Hospital Epilepsy Dataset (PhysioNet)]()
* [Kaggle Competition on Seizure Detection](https://www.kaggle.com/c/seizure-prediction/data#_=_)

## BCI
* BCI Competitions

## Emotions
* [DEAP](http://www.eecs.qmul.ac.uk/mmv/datasets/deap/) (access required)


# Visualize the Data
One quick way to look at an EDF file is via EDFBrowser. You won't be able to do any scripting, but just to look at the channels, the quality of the data, etc. It works well.

(_Insert Print Screen of EDFBrowser_)

You might encounter different file and data format (e.g. text file, csv file, edf file, etc.)
It could be a file with 1 column with values representing 1 sample every X (sampling rate) representing the amplitude(micro) volts.

# Step 3 - Preprocess the Data



# Tools
1. MNE Python
2. EEGLab
3. Brainstorm
4. EDFBrowser


# Other Datasets & Resources
http://epilepsy-database.eu/
https://physionet.org/lightwave/?db=slpdb&record=slp01a


# TODO
1. Add description of Datasets
2. Tools.