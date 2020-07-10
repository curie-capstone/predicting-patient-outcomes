# Predicting ICU Patient Mortality via ML

## Overview
The challenge is to create a model that uses data from the first 24 hours of intensive care to predict patient survival. MIT's [GOSSIS](https://gossis.mit.edu/) community initiative has provided a dataset of more than 130,000 hospital Intensive Care Unit (ICU) visits from patients, spanning a one-year timeframe. This data is part of a growing global effort and consortium spanning Argentina, Australia, New Zealand, Sri Lanka, Brazil, and more than 200 hospitals in the United States.
For this project, we're looking to improve upon the industry standard predictive model, which has been deployed by hospitals since 1992.

## Deliverables
* [Link to our website](http://curiecapstone.com/)
* [Link to our presentation](https://docs.google.com/presentation/d/1DLtEgs9hmQa4dMBxr8Yv7EFRpvykfvg2chB7D8xYb_g/edit?usp=sharing)
* [Link to our handout summarizing the project](https://github.com/curie-capstone/predicting-patient-outcomes/blob/master/icu_survived_executive_summary.pdf)

## How We Got the Data
We accessed MIT's GOSSIS data through the Women in [The Women in Data Science (WiDS) Datathon 2020 Kaggle Competition](https://www.kaggle.com/c/widsdatathon2020/overview). This data was hosted on Kaggle as a competition, and was downloaded in CSV format. Though the competition closed on February 24, 2020 the CSVs are still available from the Kaggle site. 

## Handling an Imbalanced Dataset
Our target variable, hospital_death, is a bool where 0 means the patient survived and 1 meaning the patient did not. Patients surviving account for 91.5% of the data, with the 8.5% imbalancing our data. Throughout the project our challenge was ensuring our models were applying an adequate amount of significance to the features, and not weighing solely based on the quantity of those who survived. This had a particular impact on how we handled null values within the data.

## Feature Engineering
During the project we made some features to help improve our model scores, and some of these features were very intuitive. For example, we had 3 different features for GCS (Glasglow Coma Scale) which are used for getting a measure of the alertness of a patient. We consolidated those features into a single GCS score by summing those values. But our most relevant feature was `critical_condition`, which identifies if a patient is arriving at the ICU in a critical condition and has a much higher risk of dying. This feature ended up being the first thing our models considered and put importance to when predicting patient survival.

## Modeling
For this project we used multiple classification models from the scikit-learn library including the Logistic Regression and Decision Tree algorithms. Additionally we used LGBM and XGBoost models from separate libraries. The great thing about the LGBM and XGBoost was we could feed the data directly to them without handling the missing data, allowing the model to handle the complexity of that problem how it saw fit. 

## Model Evaluation
How did we actually determine how good our model is? We're dealing with critical predictions here, so we couldn't judge our models' predictions solely on its accuracy alone. For this project we evaluated our models' performance based on the Area Under Curve (AUC) of the models predictions. How does this work? The gif below provides a visual demonstration, but the technical explanation is that an AUC chart is used to evaluate classification models by evaluating the false-positive rate and against the true-positive rate. The closer the graph moves to the upper-right, the more accurate the model is. 

![Area Under Curve](resources/images/roc.gif)


## Take Aways
Beyond providing better predictions than what is used in hospitals today, our can also provide per patient risk assessments. This gives physicians and clinicians the ability to visualize underlying risk factors for patients and better analyze where they’re headed.This model can also be used in the identification of two key patient groups, high risk patients and patients with unexpected outcomes. This means that in cases where ICU space is low and the criticality of the patients is a deciding factor on who is seen, our model can help make that determination in a timely manner. It also means that in the cases where a patient had an unexpected outcome (such as having a low predicted chance of dying but still dying anyhow) we can highlight that case and learn from it to improve the medical industry moving forward