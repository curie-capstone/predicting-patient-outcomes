# Predicting ICU Patient Mortality via ML

## Overview
The challenge is to create a model that uses data from the first 24 hours of intensive care to predict patient survival. MIT's [GOSSIS](https://gossis.mit.edu/) community initiative has provided a dataset of more than 130,000 hospital Intensive Care Unit (ICU) visits from patients, spanning a one-year timeframe. This data is part of a growing global effort and consortium spanning Argentina, Australia, New Zealand, Sri Lanka, Brazil, and more than 200 hospitals in the United States.
For this project, we're looking to improve upon the industry standard predictive model, which has been deployed by hospitals since 1992.

## Deliverables
Link to our website:
Link to our presentation:
Link to our handout summarizing the project:

## How We Got the Data
We accessed MIT's GOSSIS data through the Women in [The Women in Data Science (WiDS) Datathon 2020 Kaggle Competition](https://www.kaggle.com/c/widsdatathon2020/overview). This data was hosted on Kaggle as a competition, and was downloaded in CSV format. Though the competition closed on February 24, 2020 the CSVs are still available from the Kaggle site. 

## Handling an Imbalanced Dataset
Our target variable, hospital_death, is a bool where 0 means the patient survived and 1 meaning the patient did not. Patients surviving account for 91.5% of the data, with the 8.5% imbalancing our data. Throughout the project our challenge was ensuring our models were applying an adequate amount of significance to the features, and not weighing solely based on the quantity of those who survived. This had a particular impact on how we handled null values within the data.

## Feature Engineering
During the project we made some features to help improve our model scores, and some of these features were very intuitive. For example, we had 3 different features for GCS (Glasglow Coma Scale) which are used for getting a measure of the alertness of a patient. We consolidated those features into a single GCS score by summing those values. But our most relevant feature was `almost_dead`, which identifies if a patient is arriving at the ICU in a critical condition and has a much higher risk of dying. This feature ended up being the first thing our models considered and put importance to when predicting patient survival.

## Modeling
For this project we used multiple classification models from the scikit-learn library including the Logistic Regression and Decision Tree algorithms. Additionally we used LGBM and XGBoost models from separate libraries. The great thing about the LGBM and XGBoost was we could feed the data directly to them without handling the missing data, allowing the model to handle the complexity of that problem how it saw fit. 

## What's Next
* Describe what we can do with our insights
* Hit home with a conclusion recapping what we've learned