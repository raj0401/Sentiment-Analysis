# **Sentiment-Analysis-on-Drug-Review-Dataset**

## Overview
The Drug Review Dataset is used in this study to grasp user evaluations on several medications for different diseases. These evaluations come in both written and numerical format. The text reviews are categorized into one of five groups based on an analysis that predicts the sentiment's polarity. The LGBM, XGBoost, catBoost, and Naive Bayes Classifier classification models have all been put to the test. The LightGBM classifier has the highest accuracy, at 75%. The highly rated medications for a certain ailment are suggested to the user based on the numerical rating.


## Dataset

https://www.kaggle.com/jessicali9530/kuc-hackathon-winter-2018

## Methodology

The initial step involved cleaning the data by removing null and empty values, opting for deletion over imputation, as imputing reviews can dilute the study's intent. The data was then sorted based on user ratings. Unique conditions were identified to form the basis of the recommendation system, and the top 10 user-preferred drugs were extracted using a weighted rating approach, with the maximum rating being 10.

TextBlob, a Python NLP library, was employed to determine the sentiment polarity of user reviews. TextBlob returns polarity and subjectivity for each review, with polarity values ranging from -1 to 1, where -1 represents a negative sentiment and 1 represents a positive sentiment. Negation words invert the polarity. TextBlob's semantic labels provide a more nuanced analysis. A correlation matrix for both cleaned and uncleaned reviews indicated that removing stopwords and using snowball stemming significantly alter sentiment, leading to a cleaning process that retains stopwords.

A weighted rating feature was added to the dataset to prioritize ratings and enhance the recommendation system’s effectiveness. A heat map of the correlation matrix was plotted to show the linear relationship between features.

For final preprocessing, label encoding was applied to convert drug names and conditions into numeric values, aiding machine learning. Despite the drawback that label encoding may imply a ranked interpretation of numbers, it was preferred over one-hot encoding to avoid the high-dimensional dataset issues and potential multicollinearity traps due to over 3,600 unique values.

Various machine learning models, including LGBM, XGBoost Classifier, and CatBoost Classifier, were trained on the extensive dataset to evaluate their capability in categorizing user ratings based on review text and uncovering areas for improvement in this study.

The target variable for prediction was 'Sentiment Rating'.


## Results

LGBM achieved an accuracy of 75.2% with a true positive (TP) count of 24,251. True negatives (TN), false negatives (FN), and false positives (FP) can be derived accordingly.
XGBoost showed an accuracy of approximately 55%, with a higher TP count of 24,390, indicating greater sensitivity.
CatBoost Classifier also had an accuracy of 55% but with a lower TP count, indicating reduced sensitivity.
Naive Bayes Classifier exhibited the lowest accuracy and was therefore not suitable for predicting user review sentiment ratings based on the drugs and conditions.
