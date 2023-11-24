#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler


# # Data Cleaning (Brendan)

# In[2]:


df_customers = pd.read_csv('olist_customers_dataset.csv')


# In[3]:


df_order_items = pd.read_csv('olist_order_items_dataset.csv')


# In[4]:


df_order_payments = pd.read_csv('olist_order_payments_dataset.csv')


# In[5]:


df_order_reviews = pd.read_csv('olist_order_reviews_dataset.csv')


# In[6]:


df_orders = pd.read_csv('olist_orders_dataset.csv')


# In[7]:


df_products = pd.read_csv('olist_products_dataset.csv')


# In[8]:


df_sellers = pd.read_csv('olist_sellers_dataset.csv')


# In[9]:


df_category_name = pd.read_csv('product_category_name_translation.csv')


# In[10]:


df_order_reviews = pd.merge(df_order_reviews, df_order_items, how='left', on='order_id')


# In[11]:


df_order_reviews = pd.merge(df_order_reviews, df_order_payments, how='left', on='order_id')


# In[12]:


df_order_reviews = pd.merge(df_order_reviews, df_orders, how='left', on='order_id')


# In[13]:


df_order_reviews = pd.merge(df_order_reviews, df_products, how='left', on='product_id')


# In[14]:


df_order_reviews = pd.merge(df_order_reviews, df_sellers, how='left', on='seller_id')


# In[15]:


df_order_reviews = pd.merge(df_order_reviews, df_category_name, how='left', on='product_category_name')


# Handle missing values: Impute or drop missing data. Encode categorical variables: Convert categorical data (like product category, seller city, etc.) into numerical values using techniques like one-hot encoding or label encoding. Scale numerical features: Normalize or standardize numerical columns to bring them to a similar scale

# In[16]:


#create a copy of the dataframe to clean 
df = df_order_reviews


# In[17]:


# Dropping columns by name
columns_to_drop = ['review_id', 'order_id','review_comment_title','review_creation_date',
                   'review_answer_timestamp','payment_sequential','product_category_name','seller_city']  
df.drop(columns=columns_to_drop, inplace=True)


# fill NaN calues 

# In[18]:


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

# In[19]:


columns_to_impute = ['product_category_name_english', 'payment_type', 'seller_state','order_status']

# Mode imputation for categorical columns
for column in columns_to_impute:
    mode_value = df[column].mode()[0]  # Calculate the mode (most frequent value) for the column
    df[column].fillna(mode_value, inplace=True)  # Fill NaN values with the mode

# Verify changes
print(df.isnull().sum())


# 2. Numeric and demension Columns Mean/Median Imputation: Replace NaN values with the mean or median of the column, depending on the distribution of the data. This helps maintain the overall distribution.

# In[20]:


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

# In[21]:


# Drop NaN values in datetime columns
df.dropna(subset=['shipping_limit_date', 'order_approved_at', 'order_delivered_customer_date','order_delivered_carrier_date',
                 'shipping_limit_date'], inplace=True)
# Verify changes
print(df.isnull().sum()) 


# One Hot Encode Categorical Data

# In[22]:


# One-hot encode categorical variables
df = pd.get_dummies(df, columns=['payment_type','order_status','product_category_name_english',
                                                             'seller_state'])


# Scale numerical and demensional data 

# In[23]:


#calculate volume of product
df['size']= df['product_length_cm']*df['product_width_cm']*df['product_height_cm']


# In[24]:


# Dropping columns by name
columns_to_drop = ['product_length_cm','product_width_cm','product_height_cm']  
df.drop(columns=columns_to_drop, inplace=True)


# In[25]:


#creating new column for how quick a delivery came
df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])
df['order_delivered_customer_date'] = pd.to_datetime(df['order_delivered_customer_date'])
df['delivery_time'] = df['order_delivered_customer_date']-df['order_purchase_timestamp']


# In[26]:


#creating a new column for how soon a delivery came before or after the expected delivery time
df['order_estimated_delivery_date'] = pd.to_datetime(df['order_estimated_delivery_date'])
df['delivery_vs_estimation'] = df['order_estimated_delivery_date']-df['order_delivered_customer_date']


# In[27]:


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


# In[28]:


# Dropping columns by name
columns_to_drop = ['order_delivered_carrier_date','order_delivered_customer_date','order_estimated_delivery_date']  
df.drop(columns=columns_to_drop, inplace=True)


# In[29]:


#scaler = StandardScaler()
#df[['price', 'freight_value','payment_installments','payment_value','product_name_lenght',
 #                 'product_description_lenght','product_photos_qty','product_weight_g','size','delivery_vs_estimation_days','delivery_time_days','order_item_id',
  # 'review_score']] = scaler.fit_transform(df[[
   #               'price', 'freight_value','payment_installments','payment_value','product_name_lenght',
    #              'product_description_lenght','product_photos_qty','product_weight_g','size','delivery_vs_estimation','delivery_time',
    #'order_item_id','review_score']])


# In[30]:


#def convert_score(score):
    #if score in [4, 5]:
        #return 1
    #else:
        #return 0

# Apply the function to the 'review_score' column in the DataFrame
#df['review_score'] = df['review_score'].apply(lambda x: convert_score(x))


# In[31]:


df['review_score'].max()


# In[32]:


# Dropping columns by name
columns_to_drop = ['product_id','review_comment_message',
                  'seller_id','shipping_limit_date','customer_id','order_purchase_timestamp','order_approved_at',
                   'seller_zip_code_prefix','delivery_time','delivery_vs_estimation']  
df.drop(columns=columns_to_drop, inplace=True)


# In[33]:


df.to_csv('double check.csv', index=False)


# In[34]:


df.head()


# # Data Understanding and Exploration (Muskaan)

# Overview of Data: Start by understanding each column and its significance.
# Descriptive Statistics: Compute summary statistics (mean, median, mode, variance, etc.) for numerical columns.
# Data Visualization: Use histograms, box plots, and scatter plots to understand distributions and relationships between variables.

# Understand the dataset: Check for missing values, data types, and distribution of review scores.
# Explore correlations: Identify relationships between columns and the review score.

# In[35]:


df.shape


# In[36]:


df.columns


# In[ ]:





# # Correlation Analysis (Muskaan)

# Correlation Matrix: Calculate correlations between different columns to understand relationships.
# Correlation with Review Score: Identify columns with strong correlations to the review_score.

# In[37]:


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


# In[38]:


# Calculate the correlation matrix
correlation_matrix = df.corr()

# Calculate correlation with 'review_score'
correlation_with_review_score = correlation_matrix['review_score'].sort_values(key=abs, ascending=False)

# Remove 'review_score' itself from the correlations
correlation_with_review_score = correlation_with_review_score.drop('review_score')

# Display top 10 correlations by absolute value
top_10_correlations = correlation_with_review_score.head(10)
print(top_10_correlations)


# # Association Rule 

# In[39]:


pip install mlxtend


# In[40]:


from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules


# In[41]:


df_order_reviews.columns


# In[42]:


# Replace 'data' with your actual DataFrame and select columns relevant for analysis
# Example selection of columns:
relevant_columns = ['price', 'freight_value', 'payment_installments', 'product_category_name_english', 'review_score','payment_type',
                    'seller_state','product_photos_qty','product_description_lenght','product_name_lenght']
data_subset = df_order_reviews[relevant_columns].copy()


# In[43]:


# Convert categorical variables to one-hot encoded format
columns_to_encode = ['product_category_name_english', 'payment_type', 'seller_state']
data_encoded = pd.get_dummies(data_subset, columns=columns_to_encode)


# In[44]:


# Example code to create interval-type columns
data_encoded['price_bins'] = pd.cut(data_encoded['price'], bins=5)
data_encoded['freight_bins'] = pd.cut(data_encoded['freight_value'], bins=5)
data_encoded['payment_bins'] = pd.cut(data_encoded['payment_installments'], bins=5)
data_encoded['product_name_lenght'] = pd.cut(data_encoded['product_name_lenght'], bins=5)
data_encoded['product_description_lenght'] = pd.cut(data_encoded['product_description_lenght'], bins=5)
data_encoded['product_photos_qty'] = pd.cut(data_encoded['product_photos_qty'], bins=5)


# In[45]:


# Drop the original numerical columns
data_encoded.drop(['price', 'freight_value', 'payment_installments','product_name_lenght','product_description_lenght','product_photos_qty'], axis=1, inplace=True)


# In[46]:


# Custom function to encode categorical variables
def encode_category(x):
    if x < 3:
        return 0
    else:
        return 1

# Specify columns to encode
columns_to_encode = ['price_bins', 'freight_bins', 'payment_bins']

# Apply the encoding function to specific columns
data_encoded[columns_to_encode] = data_encoded[columns_to_encode].apply(lambda x: x.cat.codes.apply(encode_category))


# In[47]:


# Custom function to encode review_score
def encode_review_score(x):
    if x < 4:
        return 0
    else:
        return 1

# Apply the encoding function to the review_score column
data_encoded['review_score'] = data_encoded['review_score'].apply(encode_review_score)


# In[48]:


# Apply Apriori algorithm to find frequent itemsets
frequent_itemsets = apriori(data_encoded, min_support=0.05, use_colnames=True)


# Support: This measures how frequently an itemset appears in the dataset. min_support is the minimum threshold set for the support value. For example, min_support=0.01 means that an itemset must appear in at least 1% of the transactions.

# In[49]:


# Generate association rules
rules = association_rules(frequent_itemsets, metric='lift', min_threshold=1.1)


# Lift: This indicates the strength of a rule over randomness. A lift value greater than 1 suggests a stronger association.

# In[50]:


rules


# In[51]:


# Filter rules where 'review_score' is in the consequents
review_score_rules = rules[rules['consequents'].apply(lambda x: 'review_score' in str(x))]


# In[52]:


review_score_rules


# In[53]:


# Generate association rules with negative lift as well
rules_negative_lift = association_rules(frequent_itemsets, metric='lift', min_threshold=0, support_only=False)


# In[54]:


# Filter rules where 'review_score' is in the consequents
review_score_rules = rules_negative_lift[rules_negative_lift['consequents'].apply(lambda x: 'review_score' in str(x))]


# In[55]:


review_score_rules


# Antecedent: This refers to the items, conditions, or variables that appear before the arrow in an association rule (the "if" part). For instance, in the rule "if {bread, milk} then {eggs}," the antecedent is {bread, milk}. It represents the items or conditions that are considered as the basis for predicting or inferring the consequent.
# 
# Consequent: This represents the outcome, result, or items that are predicted or inferred based on the presence or occurrence of the antecedent. In the rule "if {bread, milk} then {eggs}," the consequent is {eggs}. It indicates what is likely to happen or be observed given the presence or occurrence of the antecedent.
# 
# Support: This measures how frequently a particular itemset (combination of items) appears in the dataset. It signifies the proportion of transactions in which the itemset occurs. Higher support values indicate that the itemset is more frequently observed.
# 
# Confidence: Confidence measures the reliability or strength of the association between the antecedent and consequent in a rule. It signifies the likelihood that the consequent will occur when the antecedent is present. Higher confidence values indicate a stronger relationship between the antecedent and consequent.
# 
# Higher Confidence: Values closer to 1 indicate stronger relationships between the antecedent and consequent. For instance, a confidence of 0.8 or higher might be considered high, suggesting that the consequent is frequently found in transactions where the antecedent is present.
# 
# Moderate Confidence: Values around 0.5 to 0.7 might be considered moderate and could still provide valuable insights, indicating moderate association between the antecedent and consequent.
# 
# Lower Confidence: Values closer to 0 suggest weak associations, implying that the consequent doesn't necessarily follow when the antecedent is present.
# 
# Lift: Lift indicates the strength of a rule over randomness. It measures how much more likely the consequent is, given the presence of the antecedent, compared to its likelihood without the antecedent. A lift value greater than 1 implies that the occurrence of the antecedent increases the probability of the consequent compared to its usual occurrence, while a lift less than 1 indicates a decrease in likelihood.

# Leverage: It measures the difference between the observed frequency of A and C appearing together and the frequency that would be expected if A and C were independent. It shows how much more the antecedent and consequent co-occur compared to what would be expected if they were independent.
# 
# Conviction: It calculates the ratio of the expected frequency that A occurs without C if they were independent, divided by the observed frequency of A not occurring given that C has occurred. High conviction values indicate strong association between the antecedent and consequent.
# 
# Zhang's metric: This is a specific measure used in association rule mining that considers both confidence and leverage. It's a combined metric that provides a balanced view of the association between antecedents and consequents in a rule. Higher values indicate stronger relationships.

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




