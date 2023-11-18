#!/usr/bin/env python
# coding: utf-8

# In[108]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler


# # Data Cleaning (Brendan)

# In[109]:


df_customers = pd.read_csv('olist_customers_dataset.csv')


# In[110]:


df_order_items = pd.read_csv('olist_order_items_dataset.csv')


# In[111]:


df_order_payments = pd.read_csv('olist_order_payments_dataset.csv')


# In[112]:


df_order_reviews = pd.read_csv('olist_order_reviews_dataset.csv')


# In[113]:


df_orders = pd.read_csv('olist_orders_dataset.csv')


# In[114]:


df_products = pd.read_csv('olist_products_dataset.csv')


# In[115]:


df_sellers = pd.read_csv('olist_sellers_dataset.csv')


# In[116]:


df_category_name = pd.read_csv('product_category_name_translation.csv')


# In[117]:


df_order_reviews = pd.merge(df_order_reviews, df_order_items, how='left', on='order_id')


# In[118]:


df_order_reviews = pd.merge(df_order_reviews, df_order_payments, how='left', on='order_id')


# In[119]:


df_order_reviews = pd.merge(df_order_reviews, df_orders, how='left', on='order_id')


# In[120]:


df_order_reviews = pd.merge(df_order_reviews, df_products, how='left', on='product_id')


# In[121]:


df_order_reviews = pd.merge(df_order_reviews, df_sellers, how='left', on='seller_id')


# In[122]:


df_order_reviews = pd.merge(df_order_reviews, df_category_name, how='left', on='product_category_name')


# Handle missing values: Impute or drop missing data. Encode categorical variables: Convert categorical data (like product category, seller city, etc.) into numerical values using techniques like one-hot encoding or label encoding. Scale numerical features: Normalize or standardize numerical columns to bring them to a similar scale

# In[123]:


#create a copy of the dataframe to clean 
df = df_order_reviews


# In[125]:


df.to_csv('data cleaning.csv', index=False)


# In[126]:


# Dropping columns by name
columns_to_drop = ['review_id', 'order_id','review_comment_title','review_comment_message','review_creation_date',
                   'review_answer_timestamp','payment_sequential','product_category_name']  
df.drop(columns=columns_to_drop, inplace=True)


# fill NaN calues 

# In[127]:


# Check for missing values in each column
columns_with_na = df.columns[df_order_reviews.isnull().any()]

# Display the columns with missing values
print(columns_with_na)


# 
# 
# 
# Impute with Specific Value: For date columns, you might impute NaNs with a default value (e.g., 'Not Available', 'Missing') if it's appropriate for your analysis.
# 4. Textual Columns (e.g., product_id, seller_id)
# Fill with Placeholder: For categorical identifiers, replace NaN values with a placeholder ('Not Available', 'Missing') to maintain the integrity of the dataset.
# 
# Tips:
# Consider Column Importance: Prioritize handling NaNs in columns crucial for your analysis.
# Domain Knowledge: Leverage your understanding of the data and domain to impute missing values sensibly.
# Explore Patterns: Analyze if there's a pattern behind missing values in specific columns.

# 1. Categorical Columns 
# Mode Imputation: Fill NaN values with the mode of the respective column.

# In[128]:


columns_to_impute = ['product_category_name_english', 'payment_type', 'seller_city', 'seller_state','order_status']

# Mode imputation for categorical columns
for column in columns_to_impute:
    mode_value = df[column].mode()[0]  # Calculate the mode (most frequent value) for the column
    df[column].fillna(mode_value, inplace=True)  # Fill NaN values with the mode

# Verify changes
print(df.isnull().sum())


# 2. Numeric and demension Columns Mean/Median Imputation: Replace NaN values with the mean or median of the column, depending on the distribution of the data. This helps maintain the overall distribution.

# In[129]:


numeric_columns = ['price', 'freight_value', 'product_weight_g', 'product_photos_qty','product_name_lenght',
                   'product_description_lenght','product_photos_qty','product_length_cm','product_height_cm','product_width_cm',
                   'payment_installments','payment_value']

# Check distribution and impute with mean or median accordingly
for column in numeric_columns:
    # Check skewness or other factors to determine the imputation method
    if df[column].skew() > 1:  # Example: consider skewness threshold as 1
        median_value = df[column].median()  
        df[column].fillna(median_value, inplace=True)  # Impute with median for skewed data
    else:
        mean_value = df[column].mean()
        df[column].fillna(mean_value, inplace=True) 

# Verify changes
print(df.isnull().sum()) 


# 3. Date/Time Columns
# Forward/Backward Fill: If the data has a time series nature, consider forward or backward filling NaN values based on chronological order.

# I considered doing forward fill or back fill but don't like this method since it doesn't seem accurate with our dataset. I'll drop rows instead since it is a small number of rows and we will still have a good sample size

# In[130]:


# Drop NaN values in datetime columns
df.dropna(subset=['shipping_limit_date', 'order_approved_at', 'order_delivered_customer_date','order_delivered_carrier_date',
                 'shipping_limit_date'], inplace=True)
# Verify changes
print(df.isnull().sum()) 


# One Hot Encode Categorical Data

# In[131]:


# One-hot encode categorical variables
df_order_reviews = pd.get_dummies(df_order_reviews, columns=['payment_type','order_status','product_category_name_english',
                                                             'seller_city','seller_state'])


# Scale numerical and demensional data 

# In[132]:


#calculate volume of product
df['size']= df['product_length_cm']*df['product_width_cm']*df['product_height_cm']


# In[133]:


# Dropping columns by name
columns_to_drop = ['product_length_cm','product_width_cm','product_height_cm']  
df.drop(columns=columns_to_drop, inplace=True)


# In[134]:


#creating new column for how quick a delivery came
df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])
df['order_delivered_customer_date'] = pd.to_datetime(df['order_delivered_customer_date'])
df['delivery_time'] = df['order_delivered_customer_date']-df['order_purchase_timestamp']


# In[135]:


#creating a new column for how soon a delivery came before or after the expected delivery time
df['order_estimated_delivery_date'] = pd.to_datetime(df['order_estimated_delivery_date'])
df['delivery_vs_estimation'] = df['order_estimated_delivery_date']-df['order_delivered_customer_date']


# In[136]:


# Dropping columns by name
columns_to_drop = ['order_delivered_carrier_date','order_delivered_customer_date','order_estimated_delivery_date',]  
df.drop(columns=columns_to_drop, inplace=True)


# In[101]:


scaler = StandardScaler()
df[['price', 'freight_value','payment_installments','payment_value','product_name_lenght',
                  'product_description_lenght','product_photos_qty','product_weight_g','size']] = scaler.fit_transform(df[[
                  'price', 'freight_value','payment_installments','payment_value','product_name_lenght',
                  'product_description_lenght','product_photos_qty','product_weight_g','size']])


# In[139]:


#change review_scores columns to 0 and 1 for good and bad
# Function
def convert_score(score):
    if score in [4, 5]:
        return 1
    else:
        return 0

#apply function to df
df['review_score'] = df['review_score'].apply(lambda x: convert_score(x))


# # Data Understanding and Exploration (Muskaan)

# Overview of Data: Start by understanding each column and its significance.
# Descriptive Statistics: Compute summary statistics (mean, median, mode, variance, etc.) for numerical columns.
# Data Visualization: Use histograms, box plots, and scatter plots to understand distributions and relationships between variables.

# Understand the dataset: Check for missing values, data types, and distribution of review scores.
# Explore correlations: Identify relationships between columns and the review score.

# In[38]:


df.shape


# In[39]:


df.columns


# In[40]:


df.head()


# # Correlation Analysis (Muskaan)

# Correlation Matrix: Calculate correlations between different columns to understand relationships.
# Correlation with Review Score: Identify columns with strong correlations to the review_score.

# # Hypothesis Testing (Brendan)

# Formulate Hypotheses: Based on initial observations, create hypotheses about which columns might influence review scores.
# Statistical Tests: Use appropriate statistical tests (t-tests, ANOVA, chi-square, etc.) to test these hypotheses.

# # Feature Importance (Metzel)

# Identify Important Features: Use statistical techniques (ANOVA, correlation coefficients, etc.) to determine which features are most important in predicting review scores.

# # Sentiment Analysis (Brendan)

# # Time Analysis (Muskaan)

# Temporal Trends: Check if review scores vary over time. Analyze patterns based on review_creation_date, order_purchase_timestamp, etc.

# # Multivariate Analysis (Metzel)

# Regression Analysis: Run regression models to understand how different features collectively impact review scores.

# # Report Preparation (TBD, will divide later)

# Once we are dont this we can go begin report preperation

# In[ ]:




