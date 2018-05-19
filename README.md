# dl-eeg-playground
You have found the Deep Learning EEG Playground, put together by the Montreal Hacknight. 

The repo is a bit messy, but what you should find in here:
- examples on how to usual stuff with colab
- a pyRiemann comparative example
- brain-decode based experimentations:
    - tutorial from their website
    - x86 execution
    - colab based code
    - sklearn wrapper
    
We are currently working toward integrating braindecode into MOABB, feel free to join us every other Fridays @ District 3 Innovation Center


# SETUP

- We assume you are using Anaconda, python 3.5

- Install Brain Decode: https://github.com/robintibor/braindecode
  - If you are on Windows, [You can install PyTorch using these instructions. You only need to go up to step 4.A.](https://www.superdatascience.com/pytorch/) 
- Go through the TrialWise Tutorial: https://robintibor.github.io/braindecode/notebooks/TrialWise_Decoding.html to make sure everything is setup properly
- Download this dataset : Two class motor imagery (002-2014) at http://bnci-horizon-2020.eu/database/data-sets
- put everything in an new folder (here)/BBCIData/

The remaining of the project is described in our jupyter notebook(s)
1 - Two-Classes Classification (BNCI)

# SETUP (Google Colab)

1. Download `2 - Two-Classes Classification (BNCI) Colab.ipynb` and upload it on your Google Drive.
1. Open [Google Colab](https://colab.research.google.com)
1. Instructions for dataset download and python library installation are described in the Jupyter notebook.

## Resources
For more papers on DL applications on EEG, you could refer to this [repo](https://github.com/arnaghosh/DL-neuro_Papers). Feel free to contribute.
