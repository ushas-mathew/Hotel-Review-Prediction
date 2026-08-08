# Hotel Review Sentiment Analysis Using NLP and Machine Learning

## Project Overview

This project focuses on analysing hotel reviews using Natural Language Processing (NLP), machine learning and neural network techniques.

The objective is to analyse customer reviews from TripAdvisor and classify the sentiment expressed in the reviews. The project also explores patterns in hotel reviews, reviewer characteristics and hotel ratings to identify insights that can help hospitality businesses understand customer satisfaction and areas for improvement.

Two text representation techniques, Bag-of-Words (BoW) and TF-IDF, were investigated and compared across multiple machine learning and neural network models.

## Problem Statement

Hotels receive a large volume of customer reviews containing valuable information about guest experiences, satisfaction and areas of concern.

Manually analysing thousands of reviews is time-consuming and difficult to scale. This project aims to automate the analysis of hotel reviews using NLP and machine learning techniques to identify sentiment and provide useful insights for hospitality businesses.

## Project Objectives

The main objectives of this project are:

- Analyse hotel reviews and customer ratings.
- Perform exploratory data analysis on hotel review data.
- Clean and preprocess textual review data.
- Apply Natural Language Processing techniques to review text.
- Convert textual data into numerical representations using Bag-of-Words and TF-IDF.
- Build and compare multiple machine learning models.
- Develop an Artificial Neural Network for sentiment classification.
- Evaluate model performance using accuracy, precision, recall and F1-score.
- Compare the performance of BoW and TF-IDF representations.
- Identify useful insights from customer reviews for the hospitality industry.

## Dataset

The project uses the **515k Hotel Reviews Data in Europe** dataset sourced from Kaggle.

The report describes the dataset as containing approximately 50,000 records and 17 features covering hotel reviews from Europe.

The dataset contains information relating to:

- Hotel information
- Hotel address
- Hotel location
- Reviewer nationality
- Review date
- Reviewer score
- Review text
- Review tags
- Hotel average score
- Number of reviews
- Reviewer activity
- Geographical coordinates

The review data covers the period from January 2015 to December 2017.

## Exploratory Data Analysis

Exploratory Data Analysis was performed to understand the characteristics of the hotel reviews and identify useful patterns.

The analysis included:

- Descriptive statistics
- Hotel locations
- Top-reviewed hotels
- Reviewer nationality
- Reviewer scores
- Positive review analysis
- Negative review analysis
- Word-cloud visualisation

### Key Findings

The analysis identified concentrations of highly rated hotels around major tourist locations, particularly in cities such as Paris and London.

The reviewer nationality analysis showed that the United Kingdom represented one of the largest groups of reviewers in the dataset.

Positive and negative word clouds were also generated to identify frequently occurring terms associated with positive and negative guest experiences.

Common positive terms included words such as:

- Excellent
- Helpful
- Clean
- Friendly
- Great
- Lovely

Negative reviews contained terms associated with issues such as:

- Poor service
- Dirty
- Small
- Disappointing breakfast
- Bad

These patterns can provide useful information for identifying strengths and weaknesses in hotel services.

## Data Preprocessing

The following preprocessing and feature engineering techniques were considered and applied as part of the project:

- Missing-value handling
- Data cleaning
- Text cleaning
- Removal of unnecessary characters and formatting
- Tokenisation
- Stop-word removal
- Stemming or lemmatisation
- Feature engineering
- Categorical variable encoding
- Numerical feature normalisation

These steps prepared the review data for NLP and machine learning analysis.

## Natural Language Processing

Two main text representation techniques were investigated:

### Bag-of-Words

The Bag-of-Words approach converts review text into a numerical representation based on word frequency.

The vocabulary is learned from the training data and each review is represented as a vector of word frequencies.

### TF-IDF

TF-IDF (Term Frequency-Inverse Document Frequency) represents the importance of words within individual reviews relative to the complete collection of reviews.

Both representations were used to train and compare different machine learning models.

## Machine Learning Models

The following models were implemented and evaluated:

- Logistic Regression
- Naive Bayes
- XGBoost
- Artificial Neural Network (ANN)

An 80:20 train-test split was used for model evaluation.

## Model Performance

### Bag-of-Words Results

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.81 | 0.82 | 0.79 | 0.80 |
| Naive Bayes | 0.80 | 0.82 | 0.76 | 0.79 |
| XGBoost | 0.80 | 0.81 | 0.78 | 0.80 |
| ANN | 0.817* | - | - | - |

### TF-IDF Results

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.82 | 0.81 | 0.83 | 0.82 |
| Naive Bayes | 0.81 | 0.80 | 0.83 | 0.82 |
| XGBoost | 0.79 | 0.77 | 0.79 | 0.79 |
| ANN | 0.8152* | - | - | - |

*ANN accuracy refers to validation accuracy reported in the project.

## Hyperparameter Tuning

Hyperparameter tuning was performed to investigate whether model performance could be improved.

The project explored parameter optimisation for:

- Logistic Regression
- XGBoost
- Artificial Neural Network

For the ANN, different activation functions, network structures and numbers of training epochs were investigated.

The final ANN configuration used a Softmax activation function and two dense layers.

The project also observed that increasing the number of epochs could lead to overfitting, while reducing the number of epochs improved generalisation.

## Results and Comparison

The results showed that the performance of the models varied depending on the text representation technique.

For the reported results:

- TF-IDF Logistic Regression achieved an accuracy of 82%.
- TF-IDF Naive Bayes achieved an accuracy of 81%.
- TF-IDF XGBoost achieved an accuracy of 79%.
- The ANN achieved validation accuracy of approximately 81.5% with TF-IDF.
- The ANN achieved validation accuracy of approximately 81.7% with Bag-of-Words.

Overall, the Artificial Neural Network showed strong performance, while Logistic Regression and Naive Bayes also produced competitive results.

The project also compared computational performance between BoW and TF-IDF representations.

## Business Insights

Sentiment analysis of hotel reviews can provide useful information for hospitality businesses.

Potential applications include:

- Identifying common customer complaints.
- Understanding aspects of the hotel experience that guests appreciate.
- Monitoring customer satisfaction.
- Identifying areas requiring service improvement.
- Supporting online reputation management.
- Helping hotel managers make data-driven decisions.
- Understanding customer preferences across different locations and demographics.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- Natural Language Processing (NLP)
- Jupyter Notebook
- Artificial Neural Networks

## Project Structure

```text
hotel-review-prediction/
|
├── README.md
│
├── data/
│   └── hotel_reviews.csv
│
├── notebooks/
│   └── hotel_review_prediction.ipynb
│
└── reports/
    └── hotel_review_prediction.pdf
