#!/usr/bin/env python
# coding: utf-8

# # Predicting sales using linear regression
# 
# This notebook presents a solution to a regression problem that can be found at: https://datahack.analyticsvidhya.com/contest/practice-problem-big-mart-sales-iii/
# 
# This is my first data science project, and I have built this notebook in order to develop my skills, and get myself familiar with the data science pipeline.
# 
# The algorithm used is linear regression.

# In[2]:


import pandas as pd
import numpy as np


# ## 1. Importing data and exploring the dataset

# In[3]:


raw = pd.read_csv('Documents/Data Science/Datasets/Big Mart Sales Train')


# In[4]:


raw.head()


# In[5]:


raw.info()


# In[6]:


variables = raw.drop(columns = ['Item_Identifier', 'Outlet_Identifier'])


# In[7]:


variables.Item_Fat_Content.unique()


# ### 1.1 Recoding variables
# 
# We can see that column Item_Fat_Content has diferent values for same categories (reg should be Regular for example). This means that values need to be recoded.

# In[8]:


variables = variables.replace({'Item_Fat_Content':{'low fat':'Low Fat','LF':'Low Fat','reg':'Regular'}});
variables.Item_Fat_Content.unique()


# In[9]:


variables.Item_Type.unique()


# In[10]:


variables.Outlet_Size.unique()


# In[11]:


variables.Outlet_Location_Type.unique()


# In[12]:


variables.Outlet_Type.unique()


# In[13]:


variables.describe()


# ### 1.2 Recoding variables 2
# 
# Since it doesn't make sense for the column Item_Visibility to have values of 0, every 0 is being replaced with nan.

# In[14]:


variables = variables.replace({'Item_Visibility' : {0 : np.nan}})
variables.describe()


# ### 2. Imputing missing values

# In[15]:


from autoimpute.imputations import SingleImputer


# In[16]:


raw.columns


# In[17]:


imputer = SingleImputer(strategy = {'Item_Weight':'interpolate',
                                      'Item_Visibility':'interpolate',
                                     'Outlet_Size':'categorical'}, seed = 101)


# In[18]:


#Joining Item and Outlet identifiers to imputed dataset

imputations = pd.concat([imputer.fit_transform(variables), raw.Item_Identifier, raw.Outlet_Identifier], axis = 1)


# ### 3. EDA on imputed dataset

# In[19]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[20]:


imputations.describe()


# #### 3.1 Using box plots and value counts to determine the way data is distributed

# In[21]:


fig, ax = plt.subplots(1, 4)
ax[0].boxplot(imputations.Item_Weight)
ax[0].set_title('Item Weight')
ax[1].boxplot(imputations.Item_Visibility)
ax[1].set_title('Item Visibility')
ax[2].boxplot(imputations.Item_MRP)
ax[2].set_title('Item MRP')
ax[3].boxplot(imputations.Item_Outlet_Sales)
ax[3].set_title('Item Outlet Sales')
plt.tight_layout(w_pad = 5)
plt.show()


# In[22]:


imputations.Item_Fat_Content.value_counts()


# In[23]:


imputations.Item_Type.value_counts()


# In[24]:


imputations.Outlet_Establishment_Year.value_counts()


# In[25]:


imputations.Outlet_Size.value_counts()


# In[26]:


imputations.Outlet_Location_Type.value_counts()


# In[27]:


imputations.Outlet_Type.value_counts()


# #### 3.2 Using scatter plots, Pearson coefficient and chi square test to determine relationships in data

# In[28]:


fig, ax = plt.subplots()
ax.scatter(imputations.Item_Weight, imputations.Item_Outlet_Sales)
plt.show()


# In[29]:


fig, ax = plt.subplots()
ax.scatter(imputations.Item_Visibility, imputations.Item_Outlet_Sales)
plt.show()


# In[30]:


fig, ax = plt.subplots()
ax.scatter(imputations.Item_MRP, imputations.Item_Outlet_Sales)
plt.show()


# In[31]:


from scipy import stats


# In[32]:


for column in [imputations.Item_Weight, imputations.Item_Visibility, imputations.Item_MRP]:
    print(stats.pearsonr(column, imputations.Item_Outlet_Sales))


# In[33]:


print(stats.pearsonr(imputations.Item_Weight, imputations.Item_Visibility))
print(stats.pearsonr(imputations.Item_Weight, imputations.Item_MRP))
print(stats.pearsonr(imputations.Item_Visibility, imputations.Item_MRP))


# In[34]:


print('Chi Square and p-values: ')
print('Item Fat Content and Item Type:')
print(stats.chi2_contingency(pd.crosstab(imputations.Item_Fat_Content, imputations.Item_Type))[:2])
print('Item Fat Content and Outlet Size:')
print(stats.chi2_contingency(pd.crosstab(imputations.Item_Fat_Content, imputations.Outlet_Size))[:2])
print('Item Fat Content and Outlet Location Type:')
print(stats.chi2_contingency(pd.crosstab(imputations.Item_Fat_Content, imputations.Outlet_Location_Type))[:2])
print('Item Fat Content and Outlet Type:')
print(stats.chi2_contingency(pd.crosstab(imputations.Item_Fat_Content, imputations.Outlet_Type))[:2])
print('Item Type and Outlet Size:')
print(stats.chi2_contingency(pd.crosstab(imputations.Item_Type, imputations.Outlet_Size))[:2])
print('Item Type and Outlet Location Type:')
print(stats.chi2_contingency(pd.crosstab(imputations.Item_Type, imputations.Outlet_Location_Type))[:2])
print('Item Type and Outlet Type:')
print(stats.chi2_contingency(pd.crosstab(imputations.Item_Type, imputations.Outlet_Type))[:2])
print('Outlet Size and Outlet Location Type:')
print(stats.chi2_contingency(pd.crosstab(imputations.Outlet_Size, imputations.Outlet_Location_Type))[:2])
print('Outlet Size and Outlet Type:')
print(stats.chi2_contingency(pd.crosstab(imputations.Outlet_Size, imputations.Outlet_Type))[:2])
print('Outlet Location Type and Outlet Type:')
print(stats.chi2_contingency(pd.crosstab(imputations.Outlet_Location_Type, imputations.Outlet_Type))[:2])
print('Outlet Location Type and Outlet Establishment Year:')
print(stats.chi2_contingency(pd.crosstab(imputations.Outlet_Location_Type, imputations.Outlet_Establishment_Year))[:2])
print('Outlet Type and Outlet Establishment Year:')
print(stats.chi2_contingency(pd.crosstab(imputations.Outlet_Type, imputations.Outlet_Establishment_Year))[:2])
print('Outlet Size and Outlet Establishment Year:')
print(stats.chi2_contingency(pd.crosstab(imputations.Outlet_Size, imputations.Outlet_Establishment_Year))[:2])


# #### 3.3 Setting border for outliers in target variable

# In[35]:


np.percentile(imputations.Item_Outlet_Sales, 75) + 1.5*stats.iqr(imputations.Item_Outlet_Sales)


# In[36]:


outliers = imputations['Item_Outlet_Sales'] > 6501.8699


# ### 4. Feature engineering
# 
# 1. Making dummy variables for categorical data
# 2. Making the inverse of Item_Visibility column because the scatter plot for this column resembles an inverse function
# 3. Using Outlet_Establishment_Year column to create a column that represents the number of years an outlet has been open for. The year 2013 is used as a reference point because data is supposed to be from the same year.
# 4. Dropping previously determined outliers

# In[37]:


imputations['Outlet_Type_SM1'] = imputations.Outlet_Type.replace({'Supermarket Type1' : 1, 'Supermarket Type2' : 0, 'Supermarket Type3' : 0, 'Grocery Store' : 0})
imputations['Outlet_Type_SM2'] = imputations.Outlet_Type.replace({'Supermarket Type1' : 0, 'Supermarket Type2' : 1, 'Supermarket Type3' : 0, 'Grocery Store' : 0})
imputations['Outlet_Type_SM3'] = imputations.Outlet_Type.replace({'Supermarket Type1' : 0, 'Supermarket Type2' : 0, 'Supermarket Type3' : 1, 'Grocery Store' : 0})


# In[38]:


Item_Type_Groups = imputations.Item_Type.replace({'Fruits and Vegetables' : 'Food', 'Snack Foods' : 'Food', 'Frozen Foods' : 'Food', 'Dairy' : 'Food', 'Canned' : 'Food', 'Baking Goods' : 'Food', 'Meat' : 'Food', 'Breads' : 'Food', 'Starchy Foods' : 'Food', 'Breakfast' : 'Food', 'Seafood' : 'Food', 'Household' : 'Household and Hygiene', 'Health and Hygiene' : 'Household and Hygiene', 'Soft Drinks' : 'Drinks', 'Hard Drinks' : 'Drinks'})
imputations['Item_Type_Food'] = Item_Type_Groups.replace({'Food' : 1, 'Household and Hygiene' : 0, 'Drinks' : 0, 'Others' : 0})
imputations['Item_Type_HandH'] = Item_Type_Groups.replace({'Food' : 0, 'Household and Hygiene' : 1, 'Drinks' : 0, 'Others' : 0})
imputations['Item_Type_Drinks'] = Item_Type_Groups.replace({'Food' : 0, 'Household and Hygiene' : 0, 'Drinks' : 1, 'Others' : 0})
                                      


# In[39]:


imputations['Item_Fat_Content_Low'] = imputations.Item_Fat_Content.replace({'Low Fat' : 1,'Regular' : 0})
imputations['Outlet_Size_Small'] = imputations.Outlet_Size.replace({'Small' : 1, 'Medium' : 0, 'High' : 0})
imputations['Outlet_Size_Medium'] = imputations.Outlet_Size.replace({'Small' : 0, 'Medium' : 1, 'High' : 0})
imputations['Item_Visibility_inverse'] = 1/imputations.Item_Visibility
imputations['Outlet_Years_Operating'] = 2013 - imputations[['Outlet_Establishment_Year']]
imputations = imputations.drop(imputations[outliers].index)


# ### 5. Splitting data into train and test data and setting a baseline for model evaluation
# 
# Root mean square error (RMSE) is the metric used for evaluating model performance.

# In[40]:


from sklearn.model_selection import train_test_split
imputations_train, imputations_test = train_test_split(imputations, test_size = 0.3, random_state = 123)


# In[41]:


baseline_guess = np.mean(imputations_train.Item_Outlet_Sales)
def RMSE(string, y_true, y_pred):
    value = np.sqrt(np.square(np.subtract(y_true, y_pred)).mean())
    print(string + ' ' + 'Root Mean Square Error is: ' + str(value))

RMSE('Baseline', imputations_test.Item_Outlet_Sales, baseline_guess)


# ### 6. Building the model

# In[42]:


import statsmodels.api as sm
from sklearn import linear_model as lm


# In[43]:


imputations.columns


# #### 6.1 Making the broadest model possible to get a hint about which variables to use in the final model
# 
# This model will be called Model 1

# In[52]:


predictors_1 = imputations_train[['Item_Weight',
       'Item_MRP', 'Outlet_Type_SM1',
       'Outlet_Type_SM2', 'Outlet_Type_SM3', 'Item_Type_Food',
       'Item_Type_HandH', 'Item_Type_Drinks', 'Item_Visibility_inverse',
       'Outlet_Years_Operating', 'Item_Fat_Content_Low', 'Outlet_Size_Small',
       'Outlet_Size_Medium']]

predictors_1 = sm.add_constant(predictors_1)
model_1 = sm.OLS(imputations_train.Item_Outlet_Sales, predictors_1)
result_1 = model_1.fit()
print(result_1.summary())
RMSE('Model 1', imputations_train.Item_Outlet_Sales, result_1.predict(predictors_1))


# #### 6.2 Making a model with selected variables
# 
# We can see that Model 1 possibly has multicollinearity problem (which was expected based on the Pearson and chi square tests in section 3.2) and that not all of the predictor variables are significant. Because of this we will make a model that only has variables that are significant in the previous model. This model will be called Model 2

# In[55]:


predictors_2 = imputations_train[['Item_Fat_Content_Low',       
                             'Item_MRP',  
                             'Outlet_Type_SM1',
                             'Outlet_Type_SM2',
                             'Outlet_Type_SM3']]

predictors_2 = sm.add_constant(predictors_2)
model_2 = sm.OLS(imputations_train.Item_Outlet_Sales, predictors_2)
result_2 = model_2.fit()
print(result_2.summary())
RMSE('Model 2', imputations_train.Item_Outlet_Sales, result_2.predict(predictors_2))


# Model 2 has no multicollinearity and all the variables are significant. Also, the Skewness and Kurtosis metrics suggest that errors are fairly normally distributed, and Durbin-Watson suggests that there is no autocorrelation. Furthermore, Model 2 RMSE is lower than the baseline. All this suggest that this model can be used to predict sales.
# 
# Because Model 2 above doesn't include some variables (like Item_Visibility and Outlet_Size) that can be expected to have meaning in prediction of sales, another model with some of those variables is built below (Model 3).

# In[54]:


predictors_3 = imputations_train[['Item_Visibility_inverse','Outlet_Size_Small',
       'Outlet_Size_Medium', 'Item_Type_Food', 'Item_Type_HandH', 'Item_Type_Drinks']]

predictors_3 = sm.add_constant(predictors_3)
model_3 = sm.OLS(imputations_train.Item_Outlet_Sales, predictors_3)
result_3 = model_3.fit()
print(result_3.summary())
RMSE('Model 3', imputations_train.Item_Outlet_Sales, result_3.predict(predictors_3))


# Model 3 doesn't perform better than the baseline and has insignificant variables. Because of that, this model can't be used for prediction.
# 
# The final model is Model 2 because it performs the best and satisfies all assumptions of linear regression.

# ### 7. Checking for overfitting
# 
# Checking for overfitting is done using Scikit Learn library

# In[60]:


predictors_sk = imputations_train[['Item_Fat_Content_Low',  
                             'Item_MRP',  
                             'Outlet_Type_SM1',
                             'Outlet_Type_SM2',
                             'Outlet_Type_SM3']]
model_sk = lm.LinearRegression()
result_sk = model_sk.fit(predictors_sk, imputations_train.Item_Outlet_Sales)
predictors_sk_test = imputations_test[['Item_Fat_Content_Low',  
                             'Item_MRP',  
                             'Outlet_Type_SM1',
                             'Outlet_Type_SM2',
                             'Outlet_Type_SM3']]
predictions_sk_train = model_ols.predict(predictors_sk)
predictions_sk_test = model_ols.predict(predictors_sk_test)

RMSE('Model 2', imputations_train.Item_Outlet_Sales, predictions_sk_train)
RMSE('Test', imputations_test.Item_Outlet_Sales, predictions_sk_test)


# We see that RMSE is similar for both train and test datasets, which means that there is no overfitting in Model 2.

# ### 8. Interpreting the model

# In[75]:


print(predictors_sk.columns)
result_sk.coef_


# Regression coefficients lead to following conclusions:
# 1. Items with low fat content have lower sales by 83 dollars on average than items with regular fat content.
# 2. When the price of item increases by one dollar, the sales of item increase by 13.8 dollars on average. This implies that store sells mostly inelastic goods.
# 3. On average, all supermarkets have bigger sales than grocery stores, with supermarkets of type 3 making the most money, followed by supermarkets of type 1, and then type 2.

# In[ ]:




