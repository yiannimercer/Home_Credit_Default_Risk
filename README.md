# Home Credit Default Risk

While Home Credit is currently using various statistical and machine learning methods to make these predictions, they're challenging Kagglers to help them unlock the full potential of their data. Doing so will ensure that clients capable of repayment are not rejected and that loans are given with a principal, maturity, and repayment calendar that will empower their clients to be successful.

## Overview
In this project we attempt to predict their cleint's repayment abilities.  Below you can see the various techniques that were applied.  After modeling extreme amounts of feature cleaning, engineering, and model tuning, we were able to achieve a model that predicts a client's ability to repay a loan 75% correct of the time.

* Data Cleaning
  * Imputting Missing Values 
  * Label Encoding 
  * Outlier and Anamoly Detection (if applicable removing of outliers)
* Feature Engineering & Correlation Analysis
  * Polynomial Features
  * Domain Knowledge Feature Construction 
    * CREDIT_INCOME_PERCENT: the percentage of the credit amount relative to a client's income
    * NUITY_INCOME_PERCENT: the percentage of the loan annuity relative to a client's income
    * EDIT_TERM: the length of the payment in months (since the annuity is the monthly amount due
    * YS_EMPLOYED_PERCENT: the percentage of the days employed relative to the client's age
* Machine Learning Models
  * Logistic Regression 
  * Random Forest
  * Support Vector Machine
  * Light Gradient Boosting
  
## Acknowledgements
This project was developed by following along [Edward Yi. Liu's Kaggle Notebook](https://www.kaggle.com/edwardyiliu/from-data-to-features-and-classification) for the [Kaggle Competition Home Credit Group](https://www.kaggle.com/c/home-credit-default-risk/overview) sponsored.  

## Visualizations
![Alt text](https://github.com/yiannimercer/Home_Credit_Default_Risk/blob/main/Images/Figure_1.png)
![Alt text](https://github.com/yiannimercer/Home_Credit_Default_Risk/blob/main/Images/Figure_2.png)
![Alt text](https://github.com/yiannimercer/Home_Credit_Default_Risk/blob/main/Images/Figure_3.png)
![Alt text](https://github.com/yiannimercer/Home_Credit_Default_Risk/blob/main/Images/Figure_4.png)
![Alt text](https://github.com/yiannimercer/Home_Credit_Default_Risk/blob/main/Images/Figure_5.png)
![Alt text](https://github.com/yiannimercer/Home_Credit_Default_Risk/blob/main/Images/Figure_6.png)
![Alt text](https://github.com/yiannimercer/Home_Credit_Default_Risk/blob/main/Images/Figure_7.png)
![Alt text](https://github.com/yiannimercer/Home_Credit_Default_Risk/blob/main/Images/Figure_8.png)
![Alt text](https://github.com/yiannimercer/Home_Credit_Default_Risk/blob/main/Images/Figure_9.png)
![Alt text](https://github.com/yiannimercer/Home_Credit_Default_Risk/blob/main/Images/Figure_10.png)
![Alt text](https://github.com/yiannimercer/Home_Credit_Default_Risk/blob/main/Images/Figure_11.png)
![Alt text](https://github.com/yiannimercer/Home_Credit_Default_Risk/blob/main/Images/Figure_12.png)
![Alt text](https://github.com/yiannimercer/Home_Credit_Default_Risk/blob/main/Images/Figure_13.png)
![Alt text](https://github.com/yiannimercer/Home_Credit_Default_Risk/blob/main/Images/Figure_14.png)
![Alt text](https://github.com/yiannimercer/Home_Credit_Default_Risk/blob/main/Images/Figure_15.png)
![Alt text](https://github.com/yiannimercer/Home_Credit_Default_Risk/blob/main/Images/Figure_16.png)
![Alt text](https://github.com/yiannimercer/Home_Credit_Default_Risk/blob/main/Images/Figure_17.png)


