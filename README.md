# credit-risk-classification

## Overview
This project focuses on building a machine learning model to assess the creditworthiness of borrowers using historical lending data from a peer-to-peer lending services company. The goal is to predict if a borrower will be a potential default risk based on several features provided in the dataset.

## Depencies 
 - pandas 
 - numpy
 - Path
 - sklearn
 - imblearn

## Data
The dataset, `lending_data.csv` is found in the `Resources` folder. The dataset contains historical lending data with various features such as:
- Amount requested by the borrower
- Investment from peer lending
- Loan grade and subgrade
- Interest rate
- Loan term
- Annual income of the borrower

The target variable, `loan_status`, indicates the risk of the loan:
- 0: Healthy loan
- 1: High-risk loan

## Methodology
1. **Data Splitting**:
   - The dataset was divided into features (`x`) and the target label (`y`).
   - Data was further split into training and testing sets.

2. **Logistic Regression with Original Data**:
   - A logistic regression model was trained on the original data.
   - Model's performance was evaluated using metrics like balanced accuracy score, confusion matrix, and classification report.

3. **Handling Imbalanced Data**:
   - The `RandomOverSampler` from the `imblearn` library was used to oversample the minority class in the training data to address the imbalance.

4. **Logistic Regression with Resampled Data**:
   - A logistic regression model was trained on the oversampled data.
   - Model's performance was evaluated again to see improvements from handling the imbalanced data.

## Results
The logistic regression model trained on the original data achieved an accuracy of 95.20%. However, after oversampling, the balanced accuracy score improved to 99.37%.
The precision score and recall attached below further confirmed the improved performance of the model trained with oversampled data.

### Logistic Regression on Original Data:
- Balanced accuracy score: `0.9520 (95.20%)`
- Precision score for healthy loans: `1.00`
- Recall for healthy loans: `0.99`
- Precision score for high-risk loan: `0.85`
- Recall for high-risk loans: `0.91`


### Logistic Regression on Resampled Data:
- Balanced accuracy score: `0.9937 (99.37%)`
- Precision score for healthy loans: `1.00`
- Recall for healthy loans: `0.99`
- Precision score for high-risk loan: `0.84`
- Recall for high-risk loans: `0.99`


## Conclusion and Recommendations
The logistic regression model trained with oversampled data performed exceptionally well in predicting both healthy and high-risk loans. While the model excellently flags most of the high-risk loans, it tends to flag some healthy loans as high-risk more than the model with the original data. This may be an acceptable trade-off for lending companies to reduce potential defaults. However, the cost implications of false positives and fale negatives should be considered in selecting a model.  

It's recommended to consider logistic regression model trained with oversampled data if minimizing high-risks loans is the priority. It is also recommended to consider this model for deployment with periodic retraining on new data.
