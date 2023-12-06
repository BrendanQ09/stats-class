#!/usr/bin/env python
# coding: utf-8

# In[12]:


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

#read document from csv saved in documents folder 
df_customers = pd.read_csv('D:\\Users\\Owner\\Documents\\stats\\olist_customers_dataset.csv')


# In[3]:


df_order_items = pd.read_csv('D:\\Users\\Owner\\Documents\\stats\\olist_order_items_dataset.csv')


# In[4]:


df_order_payments = pd.read_csv('D:\\Users\\Owner\\Documents\\stats\\olist_order_payments_dataset.csv')


# In[5]:


df_order_reviews = pd.read_csv('D:\\Users\\Owner\\Documents\\stats\\olist_order_reviews_dataset.csv')


# In[6]:


df_orders = pd.read_csv('D:\\Users\\Owner\\Documents\\stats\\olist_orders_dataset.csv')


# In[7]:


df_products = pd.read_csv('D:\\Users\\Owner\\Documents\\stats\\olist_products_dataset.csv')


# In[8]:


df_sellers = pd.read_csv('D:\\Users\\Owner\\Documents\\stats\\olist_sellers_dataset.csv')


# In[9]:


df_category_name = pd.read_csv('D:\\Users\\Owner\\Documents\\stats\\product_category_name_translation.csv')


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

#create another copy of the dataframe to clean
df_copy_reviews = df_order_reviews

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


# Scale numerical and demensional data 

# In[22]:


#calculate volume of product
df['size']= df['product_length_cm']*df['product_width_cm']*df['product_height_cm']


# In[23]:


# Dropping columns by name
columns_to_drop = ['product_length_cm','product_width_cm','product_height_cm']  
df.drop(columns=columns_to_drop, inplace=True)


# In[24]:


#creating new column for how quick a delivery came
df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])
df['order_delivered_customer_date'] = pd.to_datetime(df['order_delivered_customer_date'])
df['delivery_time'] = df['order_delivered_customer_date']-df['order_purchase_timestamp']


# In[25]:


#creating a new column for how soon a delivery came before or after the expected delivery time
df['order_estimated_delivery_date'] = pd.to_datetime(df['order_estimated_delivery_date'])
df['delivery_vs_estimation'] = df['order_estimated_delivery_date']-df['order_delivered_customer_date']


# In[26]:


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


# In[27]:


# Dropping columns by name
columns_to_drop = ['order_delivered_carrier_date','order_delivered_customer_date','order_estimated_delivery_date']  
df.drop(columns=columns_to_drop, inplace=True)


# In[28]:


#scaler = StandardScaler()
#df[['price', 'freight_value','payment_installments','payment_value','product_name_lenght',
 #                 'product_description_lenght','product_photos_qty','product_weight_g','size','delivery_vs_estimation_days','delivery_time_days','order_item_id',
  # 'review_score']] = scaler.fit_transform(df[[
   #               'price', 'freight_value','payment_installments','payment_value','product_name_lenght',
    #              'product_description_lenght','product_photos_qty','product_weight_g','size','delivery_vs_estimation','delivery_time',
    #'order_item_id','review_score']])


# In[29]:


#def convert_score(score):
    #if score in [4, 5]:
        #return 1
    #else:
        #return 0

# Apply the function to the 'review_score' column in the DataFrame
#df['review_score'] = df['review_score'].apply(lambda x: convert_score(x))


# In[30]:


df['review_score'].max()


# In[31]:


# Dropping columns by name
columns_to_drop = ['product_id','review_comment_message',
                  'seller_id','shipping_limit_date','customer_id','order_purchase_timestamp','order_approved_at',
                   'seller_zip_code_prefix','delivery_time','delivery_vs_estimation']  
df.drop(columns=columns_to_drop, inplace=True)


# In[32]:


df.to_csv('double check.csv', index=False)


# In[33]:


df.head()


# # Data Understanding and Exploration

# Overview of Data: Start by understanding each column and its significance.
# Descriptive Statistics: Compute summary statistics (mean, median, mode, variance, etc.) for numerical columns.
# Data Visualization: Use histograms, box plots, and scatter plots to understand distributions and relationships between variables.

# Understand the dataset: Check for missing values, data types, and distribution of review scores.
# Explore correlations: Identify relationships between columns and the review score.

# In[34]:


df.shape


# In[35]:


df.columns


# # Correlation Analysis 

# Correlation Matrix: Calculate correlations between different columns to understand relationships.
# Correlation with Review Score: Identify columns with strong correlations to the review_score.

#one-hot encode for the copies i make after 
df = pd.get_dummies(df, columns=['payment_type','order_status','product_category_name_english',
                                                             'seller_state'])

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


# In[37]:


# Calculate the correlation matrix
correlation_matrix = df.corr()

# Calculate correlation with 'review_score'
correlation_with_review_score = correlation_matrix['review_score'].sort_values(key=abs, ascending=False)

# Remove 'review_score' itself from the correlations
correlation_with_review_score = correlation_with_review_score.drop('review_score')

# Display top 10 correlations by absolute value
top_10_correlations = correlation_with_review_score.head(10)
print(top_10_correlations)


# In[38]:


#tries correlation with the intention of hypothesis testing but didn't find any hypothesis
#shifting strategy to association rules 


# # Association Rule 

# In[39]:


get_ipython().system('pip install mlxtend')



# In[40]:


from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules


# In[41]:


#Create copies for future runs


# In[42]:



# In[43]:


df_copy1 = df.copy()


# In[44]:


df_copy2 = df.copy()


# # Associations with bad review categorized as 1,2,3 

# In[45]:


relevant_columns = ['price', 'freight_value', 'payment_installments', 'review_score',
                    'product_photos_qty','product_description_lenght','product_name_lenght','order_item_id','size',
                    'payment_value']
data_encoded = df[relevant_columns].copy()


# In[46]:


# Convert categorical variables to one-hot encoded format
#columns_to_encode = ['product_category_name_english', 'payment_type', 'seller_state']
#data_encoded = pd.get_dummies(data_subset, columns=columns_to_encode)


# In[47]:


# Example code to create interval-type columns
df['price_bins'] = pd.cut(df['price'], bins=5)
df['freight_bins'] = pd.cut(df['freight_value'], bins=5)
df['payment_bins'] = pd.cut(df['payment_installments'], bins=5)
df['product_name_lenght_bins'] = pd.cut(df['product_name_lenght'], bins=5)
df['product_description_lenght_bins'] = pd.cut(df['product_description_lenght'], bins=5)
df['product_photos_qty_bins'] = pd.cut(df['product_photos_qty'], bins=5)
df['order_item_id_bins'] = pd.cut(df['order_item_id'], bins=5)
df['size_bins'] = pd.cut(df['size'], bins=5)
df['payment_value_bins'] = pd.cut(df['payment_value'], bins=5)
df['product_weight_g_bins'] = pd.cut(df['product_weight_g'], bins=5)


# In[48]:


# Drop the original numerical columns
df.drop(['price', 'freight_value', 'payment_installments','product_name_lenght','product_description_lenght',
                   'product_photos_qty','order_item_id','size','payment_value','product_description_lenght',
        'product_weight_g'], axis=1, inplace=True)


# In[49]:


# Custom function to encode categorical variables
def encode_category(x):
    if x < 3:
        return 0
    else:
        return 1

# Specify columns to encode
columns_to_encode = ['price_bins', 'freight_bins', 'payment_bins','product_name_lenght_bins','product_description_lenght_bins',
                     'product_photos_qty_bins','order_item_id_bins','size_bins','payment_value_bins','product_weight_g_bins']

# Apply the encoding function to specific columns
df[columns_to_encode] = df[columns_to_encode].apply(lambda x: x.cat.codes.apply(encode_category))


# In[50]:


# Custom function to encode review_score
def encode_review_score(x):
    if x < 4:
        return 1
    else:
        return 0

# Apply the encoding function to the review_score column
df['review_score'] = df['review_score'].apply(encode_review_score)


# In[51]:


df.to_csv('test clean 3 start.csv')


# In[52]:


# Apply Apriori algorithm to find frequent itemsets
frequent_itemsets = apriori(df, min_support=0.05, use_colnames=True)


# Support: This measures how frequently an itemset appears in the dataset. min_support is the minimum threshold set for the support value. For example, min_support=0.01 means that an itemset must appear in at least 1% of the transactions.

# Lift: This indicates the strength of a rule over randomness. A lift value greater than 1 suggests a stronger association.

# In[ ]:


# Generate association rules with negative lift as well
rules_negative_lift = association_rules(frequent_itemsets, metric='lift', min_threshold=0, support_only=False)


# In[ ]:


# Filter rules where 'review_score' is in the consequents
review_score_rules = rules_negative_lift[rules_negative_lift['consequents'].apply(lambda x: 'review_score' in str(x))]


# In[ ]:


# Sorting by multiple criteria (e.g., lift, confidence, support)
review_score_rules  = review_score_rules .sort_values(by=['lift'], ascending=False)


# In[ ]:


review_score_rules = review_score_rules[review_score_rules['consequents'].apply(lambda x: x == frozenset({'review_score'}))]


# In[ ]:


#review_score_rules = review_score_rules[review_score_rules['antecedents'].apply(lambda x: x == frozenset({'product_category_name_english_bed_bath_table'}))]


# In[ ]:


review_score_rules


# In[ ]:


review_score_rules['confidence'].max()


# In[ ]:


review_score_rules.to_csv('test11.csv')


# In[ ]:

get_ipython().system('pip install networkx')

import networkx as nx
import matplotlib.pyplot as plt

# Sort rules by 'lift' and select top N rules to visualize
top_n = 6  # Change this value to select the number of top rules to display
top_rules = review_score_rules.nlargest(top_n, 'lift')

# Create a directed graph for top rules
G = nx.DiGraph()

# Add edges and weights for top rules
for idx, row in top_rules.iterrows():
    antecedent = ', '.join(row['antecedents'])
    consequent = ', '.join(row['consequents'])
    G.add_edge(antecedent, consequent, weight=row['lift'])

# Draw the graph for top rules
plt.figure(figsize=(15, 5))
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_size=1500, node_color='skyblue', font_weight='bold', font_size=8)

# Display edge weights for top rules
labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)

plt.title(f'Top {top_n} Association Rules by Lift')
plt.show()


# Antecedent: items, conditions, or variables that appear before the arrow in an association rule (the "if" part)
# 
# Consequent: This represents the outcome, result, or items that are predicted or inferred based on the presence or occurrence of the antecedent
# 
# Support: This measures how frequently a particular itemset (combination of items) appears in the dataset. 
# 
# Confidence: Confidence measures the reliability or strength of the association between the antecedent and consequent in a rule. It signifies the likelihood that the consequent will occur when the antecedent is present.
# 
# Higher Confidence: Values closer to 1 indicate stronger relationships between the antecedent and consequent. For instance, a confidence of 0.8 or higher might be considered high, suggesting that the consequent is frequently found in transactions where the antecedent is present.
# 
# Moderate Confidence: Values around 0.5 to 0.7 might be considered moderate and could still provide valuable insights
# 
# Lower Confidence: Values closer to 0 suggest weak associations, implying that the consequent doesn't necessarily follow when the antecedent is present.
# 
# Lift: Lift indicates the strength of a rule over randomness. It measures how much more likely the consequent is, given the presence of the antecedent, compared to its likelihood without the antecedent. 
# A lift value greater than 1 implies that the occurrence of the antecedent increases the probability of the consequent compared to its usual occurrence, while a lift less than 1 indicates a decrease in likelihood.

# Leverage: It measures the difference between the observed frequency of A and C appearing together and the frequency that would be expected if A and C were independent. It shows how much more the antecedent and consequent co-occur compared to what would be expected if they were independent.
# 
# Conviction: It calculates the ratio of the expected frequency that A occurs without C if they were independent, divided by the observed frequency of A not occurring given that C has occurred. High conviction values indicate strong association between the antecedent and consequent.
# 
# Zhang's metric: This is a specific measure used in association rule mining that considers both confidence and leverage. It's a combined metric that provides a balanced view of the association between antecedents and consequents in a rule. Higher values indicate stronger relationships.

# In[ ]:


relevant_columns = ['price', 'freight_value', 'payment_installments', 'review_score',
                    'product_photos_qty','product_description_lenght','product_name_lenght','order_item_id','size',
                    'payment_value']
data_encoded = df_copy1[relevant_columns].copy()


# In[ ]:


# Convert categorical variables to one-hot encoded format
#columns_to_encode = ['product_category_name_english', 'payment_type', 'seller_state']
#data_encoded = pd.get_dummies(data_subset, columns=columns_to_encode)


# In[ ]:


# Example code to create interval-type columns
df_copy1['price_bins'] = pd.cut(df_copy1['price'], bins=5)
df_copy1['freight_bins'] = pd.cut(df_copy1['freight_value'], bins=5)
df_copy1['payment_bins'] = pd.cut(df_copy1['payment_installments'], bins=5)
df_copy1['product_name_lenght_bins'] = pd.cut(df_copy1['product_name_lenght'], bins=5)
df_copy1['product_description_lenght_bins'] = pd.cut(df_copy1['product_description_lenght'], bins=5)
df_copy1['product_photos_qty_bins'] = pd.cut(df_copy1['product_photos_qty'], bins=5)
df_copy1['order_item_id_bins'] = pd.cut(df_copy1['order_item_id'], bins=5)
df_copy1['size_bins'] = pd.cut(df_copy1['size'], bins=5)
df_copy1['payment_value_bins'] = pd.cut(df_copy1['payment_value'], bins=5)
df_copy1['product_weight_g_bins'] = pd.cut(df_copy1['product_weight_g'], bins=5)


# In[ ]:


# Drop the original numerical columns
df_copy1.drop(['price', 'freight_value', 'payment_installments','product_name_lenght','product_description_lenght',
                   'product_photos_qty','order_item_id','size','payment_value','product_description_lenght',
        'product_weight_g'], axis=1, inplace=True)


# In[ ]:


# Custom function to encode categorical variables
def encode_category(x):
    if x < 3:
        return 0
    else:
        return 1

# Specify columns to encode
columns_to_encode = ['price_bins', 'freight_bins', 'payment_bins','product_name_lenght_bins','product_description_lenght_bins',
                     'product_photos_qty_bins','order_item_id_bins','size_bins','payment_value_bins','product_weight_g_bins']

# Apply the encoding function to specific columns
df_copy1[columns_to_encode] = df_copy1[columns_to_encode].apply(lambda x: x.cat.codes.apply(encode_category))


# In[ ]:


# Custom function to encode review_score
def encode_review_score(x):
    if x < 3:
        return 1
    else:
        return 0

# Apply the encoding function to the review_score column
df_copy1['review_score'] = df_copy1['review_score'].apply(encode_review_score)


# In[ ]:


# Apply Apriori algorithm to find frequent itemsets
frequent_itemsets = apriori(df_copy1, min_support=0.01, use_colnames=True)


# Support: This measures how frequently an itemset appears in the dataset. min_support is the minimum threshold set for the support value. For example, min_support=0.01 means that an itemset must appear in at least 1% of the transactions.

# Lift: This indicates the strength of a rule over randomness. A lift value greater than 1 suggests a stronger association.

# In[ ]:


# Generate association rules with negative lift as well
rules_negative_lift = association_rules(frequent_itemsets, metric='lift', min_threshold=0, support_only=False)


# In[ ]:


# Filter rules where 'review_score' is in the consequents
review_score_rules = rules_negative_lift[rules_negative_lift['consequents'].apply(lambda x: 'review_score' in str(x))]


# In[ ]:


# Sorting by multiple criteria (e.g., lift, confidence, support)
review_score_rules  = review_score_rules .sort_values(by=['lift'], ascending=False)


# In[ ]:


review_score_rules = review_score_rules[review_score_rules['consequents'].apply(lambda x: x == frozenset({'review_score'}))]


# In[ ]:


#review_score_rules = review_score_rules[review_score_rules['antecedents'].apply(lambda x: x == frozenset({'product_category_name_english_bed_bath_table'}))]


# In[ ]:


review_score_rules


# In[ ]:


review_score_rules['confidence'].max()


# In[ ]:


review_score_rules.to_csv('test12.csv')


# In[ ]:


import networkx as nx
import matplotlib.pyplot as plt

# Sort rules by 'lift' and select top N rules to visualize
top_n = 6  # Change this value to select the number of top rules to display
top_rules = review_score_rules.nlargest(top_n, 'lift')

# Create a directed graph for top rules
G = nx.DiGraph()

# Add edges and weights for top rules
for idx, row in top_rules.iterrows():
    antecedent = ', '.join(row['antecedents'])
    consequent = ', '.join(row['consequents'])
    G.add_edge(antecedent, consequent, weight=row['lift'])

# Draw the graph for top rules
plt.figure(figsize=(15, 5))
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_size=1500, node_color='skyblue', font_weight='bold', font_size=8)

# Display edge weights for top rules
labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)

plt.title(f'Top {top_n} Association Rules by Lift')
plt.show()


# In[ ]:





# In[ ]:


relevant_columns = ['price', 'freight_value', 'payment_installments', 'review_score',
                    'product_photos_qty','product_description_lenght','product_name_lenght','order_item_id','size',
                    'payment_value']
data_encoded = df_copy2[relevant_columns].copy()


# In[ ]:


# Convert categorical variables to one-hot encoded format
#columns_to_encode = ['product_category_name_english', 'payment_type', 'seller_state']
#data_encoded = pd.get_dummies(data_subset, columns=columns_to_encode)


# In[ ]:


# Example code to create interval-type columns
df_copy2['price_bins'] = pd.cut(df_copy2['price'], bins=5)
df_copy2['freight_bins'] = pd.cut(df_copy2['freight_value'], bins=5)
df_copy2['payment_bins'] = pd.cut(df_copy2['payment_installments'], bins=5)
df_copy2['product_name_lenght_bins'] = pd.cut(df_copy2['product_name_lenght'], bins=5)
df_copy2['product_description_lenght_bins'] = pd.cut(df_copy2['product_description_lenght'], bins=5)
df_copy2['product_photos_qty_bins'] = pd.cut(df_copy2['product_photos_qty'], bins=5)
df_copy2['order_item_id_bins'] = pd.cut(df_copy2['order_item_id'], bins=5)
df_copy2['size_bins'] = pd.cut(df_copy2['size'], bins=5)
df_copy2['payment_value_bins'] = pd.cut(df_copy2['payment_value'], bins=5)
df_copy2['product_weight_g_bins'] = pd.cut(df_copy2['product_weight_g'], bins=5)


# In[ ]:


# Drop the original numerical columns
df_copy2.drop(['price', 'freight_value', 'payment_installments','product_name_lenght','product_description_lenght',
                   'product_photos_qty','order_item_id','size','payment_value','product_description_lenght',
        'product_weight_g'], axis=1, inplace=True)


# In[ ]:


# Custom function to encode categorical variables
def encode_category(x):
    if x < 3:
        return 0
    else:
        return 1

# Specify columns to encode
columns_to_encode = ['price_bins', 'freight_bins', 'payment_bins','product_name_lenght_bins','product_description_lenght_bins',
                     'product_photos_qty_bins','order_item_id_bins','size_bins','payment_value_bins','product_weight_g_bins']

# Apply the encoding function to specific columns
df_copy2[columns_to_encode] = df_copy2[columns_to_encode].apply(lambda x: x.cat.codes.apply(encode_category))


# In[ ]:


# Custom function to encode review_score
def encode_review_score(x):
    if x < 2:
        return 1
    else:
        return 0

# Apply the encoding function to the review_score column
df_copy2['review_score'] = df_copy2['review_score'].apply(encode_review_score)


# In[ ]:


df_copy2.to_csv('test clean.csv')


# In[ ]:


# Apply Apriori algorithm to find frequent itemsets
frequent_itemsets = apriori(df_copy2, min_support=0.01, use_colnames=True)


# Support: This measures how frequently an itemset appears in the dataset. min_support is the minimum threshold set for the support value. For example, min_support=0.01 means that an itemset must appear in at least 1% of the transactions.

# Lift: This indicates the strength of a rule over randomness. A lift value greater than 1 suggests a stronger association.

# In[ ]:


# Generate association rules with negative lift as well
rules_negative_lift = association_rules(frequent_itemsets, metric='lift', min_threshold=0, support_only=False)


# In[ ]:


# Filter rules where 'review_score' is in the consequents
review_score_rules = rules_negative_lift[rules_negative_lift['consequents'].apply(lambda x: 'review_score' in str(x))]


# In[ ]:


# Sorting by multiple criteria (e.g., lift, confidence, support)
review_score_rules  = review_score_rules .sort_values(by=['lift'], ascending=False)


# In[ ]:


review_score_rules = review_score_rules[review_score_rules['consequents'].apply(lambda x: x == frozenset({'review_score'}))]


# In[ ]:





# In[ ]:


review_score_rules


# In[ ]:


review_score_rules['confidence'].max()


# In[ ]:


review_score_rules.to_csv('test13.csv')


# In[ ]:


import networkx as nx
import matplotlib.pyplot as plt

# Sort rules by 'lift' and select top N rules to visualize
top_n = 6  # Change this value to select the number of top rules to display
top_rules = review_score_rules.nlargest(top_n, 'lift')

# Create a directed graph for top rules
G = nx.DiGraph()

# Add edges and weights for top rules
for idx, row in top_rules.iterrows():
    antecedent = ', '.join(row['antecedents'])
    consequent = ', '.join(row['consequents'])
    G.add_edge(antecedent, consequent, weight=row['lift'])

# Draw the graph for top rules
plt.figure(figsize=(15, 5))
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_size=1500, node_color='skyblue', font_weight='bold', font_size=8)

# Display edge weights for top rules
labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)

plt.title(f'Top {top_n} Association Rules by Lift')
plt.show()


# In[ ]:





# # Rule Interpretation

# Sentiment Analysis (Brendan)

#get a dataframe of furniture reviews from uncleaned data


# # Report Preparation (TBD, will divide later)

# Once we are dont this we can go begin report preperation
df_copy_reviews.to_csv('og dataframe.csv', index=False)
df_copy_reviews 

# In[ ]:


# In[13]:


df_order_reviews.columns


# In[ ]:




