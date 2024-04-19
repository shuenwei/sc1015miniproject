# SC1015: Mini Project - Predicting Heart Disease

- Lab: FCSA
- Group: 6

Members:
  1. Goh Shuen Wei (@shuenwei)
  2. Ng Yuhang Dilon (@dillydecoded)
  3. Yang Yunle (@YYunle)

## Contributors
1. Goh Shuen Wei – EDA,  Random Forest, Neural Network
2. Ng Yuhang Dilon – EDA, Classification Tree
3. Yang Yunle – Data Preparation, Logistic Regression

## Context

Given the prevalence of heart diseases and its severity, we are interested in identifying the most optimal model and indicators to accurately predict a patient's susceptibility to developing heart diseases, allowing for early detection.

## Description
In this repository, you can find the dataset used, all Jupyter Notebooks and the presentation slides. 

The dataset used is the ‘Heart Failure Prediction Dataset’ found on Kaggle, which can be accessed at this link: https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction/data

For detailed walkthrough, you may view the source code in the following order:

1. [EDA (Exploratory data analysis)](https://github.com/shuenwei/sc1015miniproject/blob/main/EDA.ipynb)
2. [Logistic Regression](https://github.com/shuenwei/sc1015miniproject/blob/main/Logistic%20Regression.ipynb)
3. [Random Forest](https://github.com/shuenwei/sc1015miniproject/blob/main/Random%20Forest.ipynb)
4. [Neural Network](https://github.com/shuenwei/sc1015miniproject/blob/main/Neural%20Network.ipynb)

In the EDA notebook, we first cleaned the data by removing possible incorrectly recorded data. We then did a basic exploratory analysis for every predictor against the response variable (HeartDisease). This helps us to understand the relationships between the different variables and a patient's susceptibility to developing heart diseases.

In the logistic regression notebook, we tuned the hyperparameters (C and penalty) to find the most suitable value to fit into the logistic regression model after doing one hot encoding.

In the random forest notebook, we first used classification trees of different depths (3, 4, 5), followed by random forest. We also considered the impact of class imbalance.

In the neural network notebook, we used two implementations of a neural network (MLPClassifier and Keras sequential model).

One hot encoding is done in all three models. Whether the model is good / accurate is assessed by the accuracy and F1 score of the respective model on the test datasets.

## Problem Statement
How do we best predict whether a patient has heart disease or not?
1. What are the top 3 clinical predictors?
2. Which model is best in this prediction?

## Models used
  1. Logistic Regression (sklearn)
  2. Classification Tree and Random Forest (sklearn)
  3. Neural Network (MLPClassifier from sklearn and Keras Sequential Model)
  
## Conclusion
Random Forest is the best model, with the highest accuracy and F1 score. The top three variables that can be used to predict heart disease are `ST_Slope_Up`, followed by `ST_Slope_Flat` and lastly `Oldpeak`.

| Test Data |Logistic Regression|Classification Tree|Random Forest|Neural Network|
| :--- | :---: | :----: | :----: | :----: |
|Accuracy|0.879|0.866|0.879|0.857|
|F1 Score|0.872|0.853|0.873|0.849|

(Values are rounded to 3 s.f.)

## Learning Points
- One Hot Encoding
- Logistic Regression from sklearn
- GridSearchCV from sklearn
- Classification Tree and Random Forest from sklearn
- Neural Network (MLPClassifier from sklearn and Keras Sequential Model)
- Concepts of precision, recall, F1 score, sensitivity, specificity, ROC Curve
- Handling class imbalance

## References
- https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction/data
- https://archive.ics.uci.edu/dataset/45/heart+disease
- https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7905147/#:~:text=HF%20is%20extremely%20prevalent%20in,the%20United%20States%20and%20Europe
- https://www.who.int/data/gho/indicator-metadata-registry/imr-details/2380#:~:text=Rationale%3A,and%20monitoring%20glycemia%20are%20recommended
- https://www.healthhub.sg/live-healthy/diabetes-and-heart-disease
- https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
- https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
- https://keras.io/guides/sequential_model/


