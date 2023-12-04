# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# assign csv files to variables
df_order_reviews = pd.read_csv('olist_order_reviews_dataset.csv')
df_order_items = pd.read_csv('olist_order_items_dataset.csv')
df_customers = pd.read_csv('olist_customers_dataset.csv')
df_order_payments = pd.read_csv('olist_order_payments_dataset.csv')
df_orders = pd.read_csv('olist_orders_dataset.csv')
df_products = pd.read_csv('olist_products_dataset.csv')
df_sellers = pd.read_csv('olist_sellers_dataset.csv')
df_category_name = pd.read_csv('product_category_name_translation.csv')
df_geolocation = pd.read_csv('olist_geolocation_dataset.csv')


# %%

# Let's check how many how many orders and customers there are in the database.


# Grab the orders dataset and filter by order_id then calculate the number of unique values
print("Total number of orders in the database:",df_orders['order_id'].nunique())

# Do the same for the customers dataset
print("Total Number of customers:",df_customers['customer_unique_id'].nunique())

# %%

# Where are the customers from?

# Grab the cities column from the geolocation dataset. Count the values in each location then display them.
cities = df_geolocation['geolocation_city'].value_counts()
cities


# %%

# Looking at the previous length of cities, there are 8011 unique values. Lets plot the top 10 most common cities and see their values.

# Assign the most common cities to a variable
most_common_cities = cities.head(10)

# Create a barplot and display the cities
plt.figure(figsize=(12, 6))
sns.barplot(x=most_common_cities.index, y=most_common_cities.values, color='skyblue')
plt.title('Distribution of Frequencies for Each City')
plt.xlabel('City')
plt.ylabel('Frequency')
plt.xticks(rotation=90)  
plt.show()

# %%

# Lets check the status of the orders in the orders dataset

# Grab the order_status column and filter by their order_id. Check the number of unique values, filter them by descending value and display them
status = df_orders.groupby('order_status')['order_id'].nunique().sort_values(ascending=False)
status

# %%

# How many items are ordered per purchase. Lets check the min, max, mean, and standard deviation

print("Maximum order amount is:", df_order_items['order_item_id'].max())
print("Minimum order amount is:", df_order_items['order_item_id'].min())
print("Average order amount is:", df_order_items['order_item_id'].mean().round(2))
print("Standard deviation is:", df_order_items['order_item_id'].std())

# %%

# Now lets do the same thing for the price of orders

print("Maximum cost is:", df_order_items['price'].max())
print("Minimum cost is:", df_order_items['price'].min())
print("Average cost is:", df_order_items['price'].mean().round(2))
print("Median cost is:", df_order_items['price'].median())
print("Standard deviation is:", df_order_items['price'].std())

# %%

# What payment methods are people using to purchase items?

# Lets grab the payment_type column from the order payments dataset. Count the values and print them
payment_types_counts = df_order_payments['payment_type'].value_counts()

print("Number of unique payment types:", len(payment_types_counts))

# How often do the payment types appear?
print("Payment type frequencies:")
print(payment_types_counts)

# %%

# There are some large amount of payment types. Lets create a barplot to visualize the numbers
payment_types_counts = df_order_payments['payment_type'].value_counts().head(10)

plt.figure(figsize=(12, 6))
sns.barplot(x=payment_types_counts.index, y=payment_types_counts.values, color='skyblue')
plt.title('Top 10 Payment Types by Frequency')
plt.xlabel('Payment Type')
plt.ylabel('Frequency')
plt.xticks(rotation=45)  
plt.show()


# %%

# Lets check the categories dataset. Grab the english categories and count the unique values
category_names = df_category_name['product_category_name_english'].unique()

# Print the list of category names
print('Amount of categories:', len(category_names))
for category in category_names:
    print(category)

# %%

# There are review scores in the order reviews dataset. Lets check the min, max, mean, and standard deviation

print("Maximum rating is:", df_order_reviews['review_score'].max())
print("Minumum rating is:", df_order_reviews['review_score'].min())
print("Average rating is:", df_order_reviews['review_score'].mean().round(2))
print("Standard deviation is:", df_order_reviews['review_score'].std())

# %%

# Lets create a barplot to visualize the ratings

ratings = df_order_reviews['review_score'].value_counts()

plt.figure(figsize=(12, 6))
sns.barplot(x=ratings.index, y=ratings.values, color='skyblue')
plt.title('Review score by Frequency')
plt.xlabel('Ratings')
plt.ylabel('Frequency')
plt.xticks(rotation=45)  
plt.show()

# %%

# There are also review comments included. Lets count how many there are.

# Grab the review comments and count how many there are.
review_comment_count = df_order_reviews['review_comment_message'].count()
print("Total number of reviews is:", review_comment_count)

# %%

# Lets compare the the prices of purchases and what the ratings they're given

# First we need to the order reviews and order items dataset. We'll use the order_id column as a key.
merged_df = pd.merge(df_order_reviews[['order_id', 'review_score']], df_order_items[['order_id', 'price']], on='order_id')

# Now create a scatter plot and compare the two
plt.scatter(merged_df['review_score'], merged_df['price'], alpha=0.5)
plt.title('Scatter Plot: Price vs Rating')
plt.xlabel('Rating')
plt.ylabel('Price')
plt.show()

# %%


# Lets see if there is a correlation between price, ratings, and the amount of photos included in the products.

# First we'll need to merge the order reviews and order items datasets.
merged_df_order = pd.merge(df_order_reviews[['order_id', 'review_score']], df_order_items[['order_id', 'price', 'product_id']], on='order_id')

# Now we'll merge the products dataset into the previous merged dataset.
merged_df = pd.merge(merged_df_order, df_products[['product_id', 'product_photos_qty']], on='product_id')

# Create a correlation matrix to compare the columns
correlation_matrix = merged_df[['price', 'review_score', 'product_photos_qty']].corr()

# Create a heatmap for the correlation matrix
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix: Price, Review Score, and Product Photos Quantity')
plt.show()

# %%

# Lets count how many sellers there are

# Grab the seller id column and count how many unique ids there are
sellers = df_sellers['seller_id'].nunique()
print('The amount of unique sellers is:', sellers)

# %%

# Which cities have the most sellers?

# We have to merge the sellers and geolocation datasets using the zip code prefix
merged_df = pd.merge(df_sellers, df_geolocation, left_on='seller_zip_code_prefix', right_on='geolocation_zip_code_prefix', how='left')

# Lets assign the top ten cities 
seller_counts_by_city = merged_df['geolocation_city'].value_counts().head(10)

# We can plot the cities to better visualize the data
plt.figure(figsize=(12, 6))
sns.barplot(x=seller_counts_by_city.index, y=seller_counts_by_city.values, hue=seller_counts_by_city.index, palette='viridis', legend=False)
plt.title('Top 10 Cities by Number of Sellers')
plt.xlabel('City')
plt.ylabel('Number of Sellers')
plt.xticks(rotation=45)
plt.show()

# %%
