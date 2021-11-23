
**NOTE** 

This project is a work-in-progress. Data will not be published until after publication of my dissertation. You won't actually be able to run the code yourself until it is uploaded. In the meantime, I'll keep just the in-progress notebooks, scripts, and visualizations available here for public view.

**SUMMARY**

This repo contains data and code to support replication of my dissertation chapter, "Dimensions of prestige: an Analysis of the Book Review Index, 1965-2000."

The data used by the chapter is a series of .csv files derived from the scanned and OCR'ed contents of the Book Review Index. The raw data is dirty enough that a purely rule-based regex method for data cleaning and extraction was insufficient. Instead, I used a custom LSTM sequence tagger implemented in the NLP library Flair. Code for training and prediction with the LSTM is located in "extract." 

All code for generating statistics and visualizations from the chapter are located in "notebooks." Generated images are saved to "images."

The repo currently has two scripts for preprocessing: 

1) data_prep.py - this script is an older regex-based method that is no longer in use.

2) preprocess.py - this is the most up-to-date version and is based on an LSTM sequence tagger built in [Flair](https://github.com/flairNLP/flair)






