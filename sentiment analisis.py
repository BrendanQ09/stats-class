# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from gensim.models import Word2Vec


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
df = df_order_reviews
#drop rows that don't have product_category_name_english equal bed_bath_table
df = df[df['product_category_name_english'] == 'bed_bath_table']

#drop rows where review_comment_message is Nan


###perform NLP on review_comment_message
##data preprocessing
#Null Handling: Check for and handle null values if any.
df = df.dropna(subset=['review_comment_message'])
#Text Cleaning: Remove special characters, punctuation, and irrelevant symbols.
df['review_comment_message'] = df['review_comment_message'].str.replace('[^\w\s]','')
#Tokenization: Split the text into individual words or tokens.
df['review_comment_message'] = df['review_comment_message'].apply(lambda x: x.split())
#Lowercasing: Convert all text to lowercase to ensure consistency.
df['review_comment_message'] = df['review_comment_message'].apply(lambda x: [word.lower() for word in x])
#Stopwords Removal: Eliminate common words (like "the," "and," etc.) that usually don't add much meaning.
from nltk.corpus import stopwords
stop = stopwords.words('english')
#Lemmatization: Convert words to their root form (e.g., "running" to "run") so that they can be analyzed as a single item.
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
###Step 2: Feature Extraction
##Bag-of-Words (BoW): Create a matrix representing the frequency of words in the text data.
#   CountVectorizer: Convert a collection of text documents to a matrix of token counts.
df['review_comment_message'] = df['review_comment_message'].apply(lambda x: [lemmatizer.lemmatize(word) for word in x if word not in stop])
df['review_comment_message'] = df['review_comment_message'].apply(lambda x: ' '.join(x))
##TF-IDF: Convert text to numerical form by considering term frequency-inverse document frequency.
from sklearn.feature_extraction.text import TfidfVectorizer   
# Initialize TfidfVectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['review_comment_message'])

# Create DataFrame with TF-IDF features
df_tfidf = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
##Word2Vec: Generate word embeddings to represent text numerical form.
from gensim.models import Word2Vec
# Convert text data to list of tokenized sentences
tokenized_data = df['review_comment_message'].tolist()
# Create Word2Vec model
model = Word2Vec(tokenized_data, min_count=1)
# Access vocabulary using index_to_key
words = list(model.wv.index_to_key)

                                        
###Step 3: NLP Analysis
##Sentiment Analysis: Determine sentiment (positive, negative, neutral) of each comment.
from textblob import TextBlob
df['polarity'] = df['review_comment_message'].apply(lambda x: TextBlob(x).sentiment.polarity)
df['sentiment'] = df['polarity'].apply(lambda x: 'positive' if x > 0 else ('negative' if x < 0 else 'neutral'))
##Word Frequency: Identify the most common words used in the comments.
from collections import Counter
cnt = Counter()
for text in df['review_comment_message'].values:
    for word in text.split():
        cnt[word] += 1
cnt.most_common(10)
##Word Cloud: Visualize the most common words using a word cloud.
#translate the text to english
from textblob import TextBlob
def translate_to_english(text):
    if isinstance(text, list):
        translated_list = [str(TextBlob(str(item)).translate(to='en')) for item in text]
        return " ".join(translated_list)
    else:
        return str(TextBlob(str(text)).translate(to='en'))

df['review_comment_message'] = df['review_comment_message'].apply(translate_to_english)


# Apply the translation function to the 'review_comment_message' column
df['review_comment_message'] = df['review_comment_message'].apply(translate_to_english)
#word cloud
from wordcloud import WordCloud, STOPWORDS
comment_words = ' '
stopwords = set(STOPWORDS)
for val in df['review_comment_message']:
    val = str(val)
    tokens = val.split()
    for i in range(len(tokens)):
        tokens[i] = tokens[i].lower()
    for words in tokens:
        comment_words = comment_words + words + ' '
wordcloud = WordCloud(width = 800, height = 800, background_color ='white', stopwords = stopwords, min_font_size = 10).generate(comment_words)
plt.figure(figsize = (8, 8), facecolor = None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad = 0)
plt.show()
#show negative word cloud
comment_words = ' '
stopwords = set(STOPWORDS)
for val in df[df['sentiment'] == 'negative']['review_comment_message']:
    val = str(val)
    tokens = val.split()
    for i in range(len(tokens)):
        tokens[i] = tokens[i].lower()
    for words in tokens:
        comment_words = comment_words + words + ' '
wordcloud = WordCloud(width = 800, height = 800, background_color ='white', stopwords = stopwords, min_font_size = 10).generate(comment_words)
plt.figure(figsize = (8, 8), facecolor = None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad = 0)
plt.show()

#wrds found in negative word cloud: propaganda, misleading, quality, different desire


# %%
