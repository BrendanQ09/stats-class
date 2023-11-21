#!/usr/bin/env python
# coding: utf-8

# In[968]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler


# # Data Cleaning (Brendan)

# In[969]:


df_customers = pd.read_csv('olist_customers_dataset.csv')


# In[970]:


df_order_items = pd.read_csv('olist_order_items_dataset.csv')


# In[971]:


df_order_payments = pd.read_csv('olist_order_payments_dataset.csv')


# In[972]:


df_order_reviews = pd.read_csv('olist_order_reviews_dataset.csv')


# In[973]:


df_orders = pd.read_csv('olist_orders_dataset.csv')


# In[974]:


df_products = pd.read_csv('olist_products_dataset.csv')


# In[975]:


df_sellers = pd.read_csv('olist_sellers_dataset.csv')


# In[976]:


df_category_name = pd.read_csv('product_category_name_translation.csv')


# In[977]:


df_order_reviews = pd.merge(df_order_reviews, df_order_items, how='left', on='order_id')


# In[978]:


df_order_reviews = pd.merge(df_order_reviews, df_order_payments, how='left', on='order_id')


# In[979]:


df_order_reviews = pd.merge(df_order_reviews, df_orders, how='left', on='order_id')


# In[980]:


df_order_reviews = pd.merge(df_order_reviews, df_products, how='left', on='product_id')


# In[981]:


df_order_reviews = pd.merge(df_order_reviews, df_sellers, how='left', on='seller_id')


# In[982]:


df_order_reviews = pd.merge(df_order_reviews, df_category_name, how='left', on='product_category_name')


# Handle missing values: Impute or drop missing data. Encode categorical variables: Convert categorical data (like product category, seller city, etc.) into numerical values using techniques like one-hot encoding or label encoding. Scale numerical features: Normalize or standardize numerical columns to bring them to a similar scale

# In[983]:


#create a copy of the dataframe to clean 
df = df_order_reviews


# In[984]:


# Dropping columns by name
columns_to_drop = ['review_id', 'order_id','review_comment_title','review_creation_date',
                   'review_answer_timestamp','payment_sequential','product_category_name','seller_city']  
df.drop(columns=columns_to_drop, inplace=True)


# fill NaN calues 

# In[985]:


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

# In[986]:


columns_to_impute = ['product_category_name_english', 'payment_type', 'seller_state','order_status']

# Mode imputation for categorical columns
for column in columns_to_impute:
    mode_value = df[column].mode()[0]  # Calculate the mode (most frequent value) for the column
    df[column].fillna(mode_value, inplace=True)  # Fill NaN values with the mode

# Verify changes
print(df.isnull().sum())


# 2. Numeric and demension Columns Mean/Median Imputation: Replace NaN values with the mean or median of the column, depending on the distribution of the data. This helps maintain the overall distribution.

# In[987]:


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

# In[988]:


# Drop NaN values in datetime columns
df.dropna(subset=['shipping_limit_date', 'order_approved_at', 'order_delivered_customer_date','order_delivered_carrier_date',
                 'shipping_limit_date'], inplace=True)
# Verify changes
print(df.isnull().sum()) 


# One Hot Encode Categorical Data

# In[989]:


# One-hot encode categorical variables
df = pd.get_dummies(df, columns=['payment_type','order_status','product_category_name_english',
                                                             'seller_state'])


# Scale numerical and demensional data 

# In[990]:


#calculate volume of product
df['size']= df['product_length_cm']*df['product_width_cm']*df['product_height_cm']


# In[991]:


# Dropping columns by name
columns_to_drop = ['product_length_cm','product_width_cm','product_height_cm']  
df.drop(columns=columns_to_drop, inplace=True)


# In[992]:


#creating new column for how quick a delivery came
df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])
df['order_delivered_customer_date'] = pd.to_datetime(df['order_delivered_customer_date'])
df['delivery_time'] = df['order_delivered_customer_date']-df['order_purchase_timestamp']


# In[993]:


#creating a new column for how soon a delivery came before or after the expected delivery time
df['order_estimated_delivery_date'] = pd.to_datetime(df['order_estimated_delivery_date'])
df['delivery_vs_estimation'] = df['order_estimated_delivery_date']-df['order_delivered_customer_date']


# In[994]:


def extract_days(value):
    try:
        if isinstance(value, str):
            return int(value.split()[0])
        else:
            return None
    except (AttributeError, IndexError, ValueError):
        return None

# Extracting days from 'delivery_time'
df['delivery_time'] = df['delivery_time'].apply(extract_days)

# Extracting days from 'delivery_vs_estimation'
df['delivery_vs_estimation'] = df['delivery_vs_estimation'].apply(extract_days)


# In[995]:


# Dropping columns by name
columns_to_drop = ['order_delivered_carrier_date','order_delivered_customer_date','order_estimated_delivery_date']  
df.drop(columns=columns_to_drop, inplace=True)


# In[996]:


scaler = StandardScaler()
df[['price', 'freight_value','payment_installments','payment_value','product_name_lenght',
                  'product_description_lenght','product_photos_qty','product_weight_g','size','delivery_vs_estimation_days','delivery_time_days','order_item_id',
   'review_score']] = scaler.fit_transform(df[[
                  'price', 'freight_value','payment_installments','payment_value','product_name_lenght',
                  'product_description_lenght','product_photos_qty','product_weight_g','size','delivery_vs_estimation','delivery_time',
    'order_item_id','review_score']])


# In[997]:


#def convert_score(score):
    #if score in [4, 5]:
        #return 1
    #else:
        #return 0

# Apply the function to the 'review_score' column in the DataFrame
#df['review_score'] = df['review_score'].apply(lambda x: convert_score(x))


# In[998]:


df['review_score'].max()


# In[999]:


# Dropping columns by name
columns_to_drop = ['product_id','review_comment_message',
                  'seller_id','shipping_limit_date','customer_id','order_purchase_timestamp','order_approved_at',
                   'seller_zip_code_prefix','delivery_time','delivery_vs_estimation','delivery_vs_estimation_days','delivery_time_days']  
df.drop(columns=columns_to_drop, inplace=True)


# In[1000]:


df.to_csv('double check.csv', index=False)


# # Data Understanding and Exploration (Muskaan)

# Overview of Data: Start by understanding each column and its significance.
# Descriptive Statistics: Compute summary statistics (mean, median, mode, variance, etc.) for numerical columns.
# Data Visualization: Use histograms, box plots, and scatter plots to understand distributions and relationships between variables.

# Understand the dataset: Check for missing values, data types, and distribution of review scores.
# Explore correlations: Identify relationships between columns and the review score.

# In[1001]:


df.shape


# In[1002]:


df.columns


# In[ ]:





# # Correlation Analysis (Muskaan)

# Correlation Matrix: Calculate correlations between different columns to understand relationships.
# Correlation with Review Score: Identify columns with strong correlations to the review_score.

# In[1003]:


# Calculate correlation with Review Score
correlation_with_review_score = df.corr()['review_score'].sort_values(ascending=False)

# Remove 'review_score' from the correlations (it'll be perfectly correlated with itself)
correlation_with_review_score = correlation_with_review_score.drop('review_score')

# Set the minimum correlation threshold
min_threshold = 0.3
significant_correlations = correlation_with_review_score[
    abs(correlation_with_review_score) > min_threshold
]

if not significant_correlations.empty:
    # Set up the matplotlib figure
    plt.figure(figsize=(8, 6))

    # Create a horizontal bar plot for correlations above the threshold
    significant_correlations.plot(kind='barh', color='skyblue')
    plt.title(f'Correlation with Review Score (Threshold = {min_threshold})')
    plt.ylabel('Columns')
    plt.xlabel('Correlation Coefficient')
    plt.tight_layout()
    plt.show()
else:
    print("No correlations above the specified threshold.")


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




