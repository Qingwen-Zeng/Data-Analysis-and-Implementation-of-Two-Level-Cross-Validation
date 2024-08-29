# Implementation-of-PCA-and-Two-Level-Cross-Validation

This project includes the entire process of data cleaning, visualization, PCA analysis, and comparison of the results of fitting different machine learning methods for classification and regression under Two-Level-Cross-Validation.

**Data set**:Algerian_forest_fires_dataset_UPDATE.csv

**Data Analysis and PCA**: 

-Data Analysis and PCA Part1: Data cleaning, data distribution bar chart, data correlation heat map, PCA dimensionality reduction, distribution scatter plot of data on different principal components after PCA dimensionality reduction (2D, 3D)

-Data Analysis and PCA Part2: Data visualization optimization, adding Variance Explained by Each Principal Component and coefficients of the principal components

**Machine Learning Application**:

-Implementation of regularized linear regression:  regularized linear regression analysis using cross-validation on a dataset

-Regression_two-level cross validation (ANN and Ridge Regression): 

-Classification_two-level cross validation(LR and KNN ): logistic regression with L2 regularization to evaluate the effect of regularization strength on training and test error rates, and the L2 norm of the model's coefficients. K-Nearest Neighbors (KNN) classification model with cross-validation to determine the optimal number of neighbors K for classification. Use two-Level cross validation to calculate the generalization error of the two methods and compare and visualize them. McNemar's test compares the performance of two classifiers.

