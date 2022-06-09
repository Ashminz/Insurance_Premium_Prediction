# Insurance_Premium_Prediction
Link:  https://insurance-premiumprediction.herokuapp.com/


To give people an estimate of how much they need based on their individual health situation. After that, customers can work with any health insurance carrier and its plans and perks while keeping the projected cost from our study in mind. I am considering variables as age, sex, BMI, number of children, smoking habits and living region to predict the premium. This can assist a person in concentrating on the health side of an insurance policy rather than the ineffective part.

![image](https://user-images.githubusercontent.com/76057261/170078370-2b002c7f-d8f3-4079-8ca6-07d111cc8ae9.png)

# Approach

* Loading the dataset using Pandas and performed basic checks like the data type of each column and having any missing values.
Performed Exploratory data analysis:
* Visualized each predictor or independent feature with the target feature and found that there's a direct proportionality between cement and the target feature while there's an inverse proportionality between water and the target feature.
* To get even more better insights, plotted both Pearson and Spearman correlations, which showed the same results as above.
the distribution of the target feature, expenses which was in Normal distribution with a very little right skewness.
* Checked for the presence of outliers in all the columns
* Experimenting with various ML algorithms
* First, tried with Linear regression models, ridge and lasso regression approached. Performance metrics are calculated for all the approaches. The test RMSE score is little bit lesser compared to other approaches. Then, performed a residual analysis and the model satisfied all the assumptions of linear regression.
* Next, tried with various tree based models, performed hyper parameter tuning using the GridSearchCV and found the best hyperparameters for each model. Then, picked the top most features as per the feature importance by an each model. Models, evaluated on both the training and testing data and recorded the performance metrics.
* Based on the performance metrics of both the linear and the tree based models, XGBoost regressor performed the best, followed by the random forest regressor.
* Deployment: Deployed the XGBoost regressor model using Flask, which works in the backend part while for the frontend UI Web page, used HTML.

