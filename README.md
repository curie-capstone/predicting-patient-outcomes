# Predicting ICU Patient Mortality via ML

## Overview
The challenge is to create a model that uses data from the first 24 hours of intensive care to predict patient survival. MIT's [GOSSIS](https://gossis.mit.edu/) community initiative, with privacy certification from the Harvard Privacy Lab, has provided a dataset of more than 130,000 hospital Intensive Care Unit (ICU) visits from patients, spanning a one-year timeframe. This data is part of a growing global effort and consortium spanning Argentina, Australia, New Zealand, Sri Lanka, Brazil, and more than 200 hospitals in the United States.

## Where the Data is Coming From
Data is derived from [The Women in Data Science (WiDS) Datathon 2020](https://www.kaggle.com/c/widsdatathon2020/overview), which focuses on patient health through data from MIT’s GOSSIS (Global Open Source Severity of Illness Score) initiative. This data was hosted on Kaggle as a competition, and was downloaded in CSV format. Though the competition closed on February 24, 2020 the csvs are still available from the Kaggle site. The full data dictionary for the raw data can be found in the resources folder located in this project. 

## How We Tranformed the Data
In order to make full use of this data for exploring and modeling, we had to make transformations to the data to get it to the shape we needed. The primary conflicts we ran into while handling this data is the following:
- Missing values within the data
- Too many features (aka columns)

* Once we've locked down what method we're using to transform data (either the hand crafted approach or the iterative approach), describe that process here. Images will likely be a useful and fast way of explaining if possible.

## What the Data Tells Us
* Insert a bunch of stuff from explore *here*

## Can We Model Death?
* Insert results and descriptions about the model *here*
* Describe what features played the biggest impact, and hypothesis on why

## What's Next
* Describe what we can do with our insights
* Hit home with a conclusion recapping what we've learned