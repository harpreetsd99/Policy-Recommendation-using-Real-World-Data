# #### Importing Files

# In[4]:


# For Data Analysis
import pandas as pd
import numpy as np
# For Visuals
import seaborn as sns
import matplotlib.pyplot as plt
# For PDF manipulation
import requests
import os.path
import PyPDF2
import pdfreader
# For NLP
import nltk
import spacy 
nlp = spacy.load("en_core_web_sm")
from collections import defaultdict
import locationtagger
import nltk.corpus
from nltk.corpus import wordnet
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from nltk.tokenize import word_tokenize 
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
from thefuzz import fuzz
from nltk import ngrams
import textract  
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')
stopwords_eng = set(stopwords.words('english'))
nltk.downloader.download('maxent_ne_chunker')
nltk.downloader.download('words')
nltk.downloader.download('treebank')
nltk.downloader.download('maxent_treebank_pos_tagger')
nltk.downloader.download('punkt')
nltk.download('averaged_perceptron_tagger')
# Ignoring Warnings 
import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)


# In[5]:


df_policy = pd.read_excel(r"C:\\Experiential learning assignment\\City of Portland 2023 .xlsx",sheet_name='Sheet1',header=1)
df_policy=df_policy.drop(["Category Types","Issue Category "],axis=1)
df_policy


# In[6]:


df_users = pd.read_excel(r"C:\\Experiential learning assignment\\City of Portland 2023 .xlsx",sheet_name='Sheet2')
df_users = df_users.drop(["??","If loop"],axis=1)
df_users


# #### Analysis of User Sheet

# In[7]:


df_users.columns


# In[8]:


df_users.info()


# In[9]:


# Checking NA's
df_users.isna().sum()


# In[10]:


# Dropping NA's
df_users.dropna(inplace=True)


# In[11]:


df_users.isna().sum()


# In[12]:


# Changing Dtype
df_users = df_users.astype({'Infrastructure': 'int64','Education': 'int64','Zoning': 'int64', 'Safety': 'int64'})


# In[13]:


df_users.describe()


# In[14]:


df_users.info()


# In[15]:


df_users


# In[16]:


ax = sns.countplot(x="Interest ",data=df_users, order = df_users['Interest '].value_counts().index)
for p, label in zip(ax.patches, df_users['Interest '].value_counts()):
    ax.annotate(label, (p.get_x()+0.275, p.get_height()+0.55))
ax.set(xlabel='Interest', ylabel='Frequency')
plt.title("Frequency of Interests")
plt.show()


# In[17]:


plt.figure(figsize=(14,7))
ax = sns.countplot(x="AGE",data=df_users, order = df_users['AGE'].value_counts().index)
for p, label in zip(ax.patches, df_users['AGE'].value_counts()):
    ax.annotate(label, (p.get_x()+0.175, p.get_height()+0.55))
ax.set(xlabel='Age', ylabel='Frequency')
plt.title("Frequency of Interests")
plt.show()


# In[18]:


plt.figure(figsize=(14,7))
ax = sns.countplot(x="Address",data=df_users, order = df_users['Address'].value_counts().index)
for p, label in zip(ax.patches, df_users['Address'].value_counts()):
    ax.annotate(label, (p.get_x()+0.275, p.get_height()+0.55))
ax.set(xlabel='Address', ylabel='Frequency')
plt.title("Frequency of Interests")
plt.show()


# In[19]:


plt.figure(figsize=(14,7))
ax = sns.countplot(x="AGE",hue="Interest ",data=df_users)
ax.set(xlabel='Age', ylabel='Interest')
plt.title("Age Vs Interests Frequency")
plt.show()


# In[20]:


plt.figure(figsize=(14,7))
ax = sns.countplot(x="Address",hue="Interest ",data=df_users)
ax.set(xlabel='Address', ylabel='Interest')
plt.title("Address Vs Interests Frequency")
plt.show()


# In[21]:


dfm = df_users[['Environmental','Infrastructure', 'Education', 'Zoning', 'Safety']].melt()
dfm


# In[22]:


plt.figure(figsize=(12,6))
ax = sns.barplot(x="variable", y="value", data=dfm, order = dfm.groupby("variable").sum().sort_values("value",ascending=False).index)
for p, label in zip(ax.patches, dfm.groupby("variable").mean().sort_values("value",ascending=False).round(2).value):
    ax.annotate(label, (p.get_x()+0.275, p.get_height()+0.16))
ax.set(xlabel='Interests', ylabel='Values')
plt.title("Ranking Interests")
plt.show()


# In[23]:


plt.figure(figsize=(12,6))
ax=sns.boxplot(x="variable", y ="value",data=dfm)
ax.set(xlabel='Interests', ylabel='Values')
plt.title("Range of Interests")
plt.show()


# In[24]:


df_users.corr()


# In[25]:


sns.heatmap(df_users.corr())


# #### Analysis of Policy Sheet

# In[26]:


df_policy


# #### Categorising by title

# In[27]:


category=[]
def categorize_order(title):
    title = title.lower()
    categories = {
        'Environmental': ['environment', 'protection', 'brownfields', 'epa', 'parks', 'recreation'],
        'Infrastructure': ['improvement', 'transportation', 'roads', 'infrastructure', 'paving'],
        'Education': ['school', 'education', 'students', 'teachers', 'educational', 'public education'],
        'Zoning': ['zoning', 'map amendment', 'height', 'rezoning', 'land use'],
        'Safety': ['safety', 'emergency', 'firefighters', 'police', 'public safety'],
    }

    for category, keywords in categories.items():
        if any(keyword in title for keyword in keywords):
            return category

    return 'Uncategorized'

# Categorize and print the titles
for title in df_policy["Description "]:
    category.append(categorize_order(title))

category


# #### Extracting pdf

# In[28]:


'''
# Downloading file
url = df_policy["Link"][301]
r = requests.get(url, allow_redirects=True)

# Creating pdf file by writing
classification_number = df_policy["Classification Number "][1]
# Replace forward slashes with hyphens
classification_number = classification_number.replace("/", "-")
pdf_filename = f"{classification_number}.pdf"


if os.path.isfile(pdf_filename): 
    pass
else:   
    open(pdf_filename, 'wb').write(r.content)

#Reading PDf and extracting text
pdfFileObj = open(pdf_filename, 'rb')
pdfReader = PyPDF2.PdfReader(pdfFileObj)
count = len(pdfReader.pages)
output = ''
for i in range(count):
    page = pdfReader.pages[i]
    output += page.extract_text()
    
output

# Tokenization
token = word_tokenize(output)
print (token)
'''


# In[35]:


def policy_categorisation(df_policy):
    
    # Creating Loc Column
    if "Location_tag" not in df_policy.columns:
        df_policy["Location_tag"]=""
        df_policy["Location_tag"] = df_policy["Location_tag"].astype('object')
    else:
        pass
    
    # Creating Category Column
    if "Category_tag" not in df_policy.columns:
        df_policy["Category_tag"]=""
        df_policy["Category_tag"] = df_policy["Category_tag"].astype('object')
    else:
        pass
        
    #Getting Length
    length = df_policy.shape[0]
    
    #Looping
    for i in range(0,length):   
        url = df_policy["Link"][i]
        r = requests.get(url, allow_redirects=True)
        
        # Creating pdf file by writing
        classification_number = df_policy["Classification Number "][i]
        classification_number = classification_number.replace("/", "-")
        pdf_filename = f"{classification_number}.pdf"

        if os.path.isfile(pdf_filename): 
            pass
        else:   
            open(pdf_filename, 'wb').write(r.content)
            

        pdfFileObj = open(pdf_filename, 'rb')
        pdfReader = PyPDF2.PdfReader(pdfFileObj)
        count = len(pdfReader.pages)
        output=''
        for j in range(count):
           page = pdfReader.pages[j]
           output += page.extract_text() 
        
        # Tokenization
        token_words = word_tokenize(output)
        
        #Cleaning
        clean_txt = cleaned_text(token_words)
        
        #Location Tagging
        location_tags = location_tagging_(clean_txt)       
        df_policy.at[i, "Location_tag"] = location_tags
        
        #Categorisation
        category = categorization_fuzz(clean_txt)        
        df_policy.at[i, "Category_tag"] = category
    
    return df_policy


# In[36]:


df_policy_copy = df_policy.copy()
df_categorised = policy_categorisation(df_policy_copy)


# # df_categorised

# In[37]:


df_categorised["Category_tag"].value_counts()


# In[38]:


df_categorised["Category_tag"]=df_categorised["Category_tag"].replace("Uncategorized","Infrastructure")


# In[39]:


df_categorised["Location_tag"].value_counts()


# In[40]:


df_categorised.to_excel("Categorised_Policy.xlsx")


# #### Cleaning Text 

# In[29]:


# Cleaning
def  cleaned_text(tokened_txt):
    cleaned_data = []
    # Removing non-alpha text
    for w in tokened_txt:
        if w.isalpha():
            cleaned_data.append(w)
    # Removing Stopwords
    clean_words = [x for x in cleaned_data if x not in stopwords_eng]
    
    # Lemmatizer
    wordnet_lemmatizer = WordNetLemmatizer()
    lemmatized_word = [wordnet_lemmatizer.lemmatize(word) for word in clean_words]
    
    # Returning Clened Text
    return lemmatized_word


# In[63]:


#trial
cleaned_data = cleaned_text(token)
cleaned_data


# #### Extracting Location

# In[30]:


#Location Tagging
def location_tagging_(text):
    
    predefined_areas = [
    "West End",
    "Parkside",
    "East Bayside",
    "East End",
    "Bayside",
    "Deering",
    "Munjoy Hill",
    "Back Cove",
    "Old Port",
    "Highlands"]
    
    txt = " ".join(text)
    doc = nlp(txt)
    addresses = [ent.text for ent in doc.ents if ent.label_ == 'GPE']
    best_match = None
    best_score = 0
    for address in addresses:
        current_match, score = process.extractOne(address, predefined_areas)
        if score > best_score:
            best_match = current_match
            best_score = score
            
    return best_match if best_match else "None"


# In[61]:


#trial
loc_ = location_tagging_(cleaned_data)
loc_


# In[ ]:


# def location_tagging(text):
#     txt = " ".join(text)
#     place_entity = locationtagger.find_locations(text = txt)
#     return place_entity.region_cities.keys()


# #### Categorising

# In[31]:


def find_synonyms(word):
    synonyms = []
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.append(lemma.name())
    return list(set(synonyms))


# In[32]:


def categorisation(txt):
    
    categories = {
    "Environmental": ["environment", "conservation", "planet", "ecology", "sustainability", "biodiversity", "renewable resources", "climate change"],
    "Infrastructure": ["infrastructure", "construction", "roads", "utilities", "urban development", "transportation", "energy infrastructure"],
    "Education": ["education", "school", "students", "teachers", "curriculum", "educational technology", "higher education"],
    "Zoning": ["zoning", "planning", "regulations", "land use", "property development", "zoning codes", "real estate zoning"],
    "Safety": ["safety", "security", "crime", "protection", "emergency response", "disaster preparedness", "fire safety", "public safety"]}
    
    # TF-IDF
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(["".join(doc) for doc in txt])
    
    # Dictionary to store cumulative scores for each category in a document
    document_scores = defaultdict(float)

    # Categorize each tokenized document based on TF-IDF scores and synonyms
    for i, document_tokens in enumerate(txt):
        for category, keywords in categories.items():
            keyword_scores = []
            for keyword in keywords:
                synonyms = find_synonyms(keyword)
                for synonym in synonyms:
                    keyword_scores.append(tfidf_matrix[i, tfidf_vectorizer.vocabulary_.get(synonym, 0)])
            document_scores[category] += sum(keyword_scores)

    # Choose the category with the highest cumulative score for each tokenized document
    final_category = max(document_scores, key=document_scores.get)
    return final_category


# In[33]:


def categorization_fuzz(tokens):
    categories = {
        "Environmental": ["environment", "conservation", "planet", "ecology", "sustainability", "biodiversity", "renewable resources", "climate change"],
        "Infrastructure": ["infrastructure", "construction", "roads", "utilities", "urban development", "transportation", "energy infrastructure"],
        "Education": ["education", "school", "students", "teachers", "curriculum", "educational technology", "higher education"],
        "Zoning": ["zoning", "planning", "regulations", "land use", "property development", "zoning codes", "real estate zoning"],
        "Safety": ["safety", "security", "crime", "emergency", "protection", "emergency response", "disaster preparedness", "fire safety", "public safety"]
    }

    document_scores = defaultdict(float)

    for category, keywords in categories.items():
        for keyword in keywords:
            for token in tokens:
                similarity = fuzz.ratio(keyword, token)
                if similarity > 60:  # Adjust the threshold as needed
                    document_scores[category] += similarity / 100  # Normalize the score

    if document_scores:  # Check if the dictionary is not empty
        final_category = max(document_scores, key=document_scores.get)
        return final_category
    else:
        return "Uncategorized"


# In[41]:


categor = categorization_fuzz(cleaned_data)
categor


# ### Model

# In[42]:


df_categorised


# In[97]:


df_categorised.columns


# In[43]:


df_users


# In[44]:


df_users.columns


# In[48]:


customer_df=df_users.copy()
policy_df=df_categorised.copy()


# In[84]:


import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
from nearpy import Engine
from nearpy.hashes import RandomBinaryProjections

# Assuming you have already defined the Engine class

# Label encoding for Address and Interest
label_encoder_address = LabelEncoder()
label_encoder_interest = LabelEncoder()

customer_df['Address'] = label_encoder_address.fit_transform(customer_df['Address'])
customer_df['Interest'] = label_encoder_interest.fit_transform(customer_df['Interest'])

# Label encoding for Category and Location tag in policy data
label_encoder_category = LabelEncoder()
label_encoder_location = LabelEncoder()

policy_df['Category_tag'] = label_encoder_category.fit_transform(policy_df['Category_tag'])
policy_df['Location_tag'] = label_encoder_location.fit_transform(policy_df['Location_tag'])

# Merge policy and customer data
merged_df = pd.merge(customer_df, policy_df, how='cross')

# TF-IDF vectorization on policy category and location tag
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(merged_df[['Category_tag', 'Location_tag']].astype(str).agg(' '.join, axis=1))

# Get feature names from the vocabulary
feature_names = tfidf_vectorizer.get_feature_names()

# Create a sparse matrix with policy indices
policy_indices = merged_df.index.tolist()
tfidf_sparse = csr_matrix(tfidf_matrix)

# Create LSH engine
dimension = len(feature_names)
projection_count = 5  # You can adjust this value as needed
engine = Engine(dimension, lshashes=[RandomBinaryProjections('rbp', projection_count)])

# Index policy vectors
for i, idx in enumerate(policy_indices):
    vector = tfidf_sparse.getrow(i).toarray().flatten()
    engine.store_vector(vector, idx)

recommendations = []
batch_size = 100
for i in range(0, tfidf_matrix.shape[0], batch_size):
    batch_vectors = tfidf_matrix[i:i+batch_size].toarray()
    for vector in batch_vectors:
        neighbors = engine.neighbours(vector)
        indices = [n[1] for n in neighbors]
        recommendations.append([policy_indices[n] for n in indices])

print(recommendations)


# In[71]:


print(customer_df.columns)



# In[89]:


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

def recommend_policies(user_location, user_category, customer_df, policy_df):
    # Combine user's location and category tags
    user_combined_tags = f"{user_category} {user_location}"

    # Combine policy's location and category tags
    policy_df['Combined_Tags'] = policy_df[['Category_tag', 'Location_tag']].astype(str).agg(' '.join, axis=1)

    # Concatenate all tags for TF-IDF vectorization
    all_tags = list(policy_df['Combined_Tags']) + [user_combined_tags]

    # TF-IDF vectorization
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(all_tags)

    # Compute cosine similarity between user and policies
    cosine_similarities = linear_kernel(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()

    # Get indices of policies sorted by similarity
    similar_policy_indices = cosine_similarities.argsort()[::-1]

    # Get recommended policies excluding the user's input
    recommended_policies = policy_df.iloc[similar_policy_indices[1:]] 

    return recommended_policies[['Category_tag', 'Location_tag']]

# Example usage:
user_location = 'West End'  # Replace with the actual user's location
user_category = 'Enviromental'  # Replace with the actual user's category

# Assuming 'df_users' and 'df_categorised' are your user and policy dataframes
recommended_policies = recommend_policies(user_location, user_category, df_users, df_categorised)

if not recommended_policies.empty:
    print("Recommended Policies:")
    print(recommended_policies)
else:
    print("No matching policies found for the given location and category.")


# In[103]:


import pandas as pd

def recommend_policies(user_interest, user_location, user_age, customer_df, policy_df):
    # Filter policies based on user location
    relevant_policies = policy_df[policy_df['Location_tag'] == user_location]
    
    # Merge user and policy data based on the common field (e.g., 'Category_tag')
    merged_df = pd.merge(customer_df, relevant_policies, how='cross')

    # Create a combined tag for user's interest and location
    user_combined_tags = f"{user_interest} {user_location}"

    # Combine policy's interest and location tags
    policy_df['Combined_Tags'] = policy_df[['Description ', 'Location_tag']].astype(str).agg(' '.join, axis=1)

    # Concatenate all tags for TF-IDF vectorization
    all_tags = list(policy_df['Combined_Tags']) + [user_combined_tags]

    # TF-IDF vectorization
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(all_tags)

    # Calculate cosine similarity between user and policy vectors
    cosine_sim = linear_kernel(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()

    # Get indices of policies with high similarity
    similar_policies_indices = cosine_sim.argsort()[:-6:-1]

    # Extract recommended policies
    recommended_policies = policy_df.iloc[similar_policies_indices]

    if recommended_policies.empty:
        return pd.DataFrame(columns=policy_df.columns)
    else:
        return recommended_policies

# Example of usage
user_interest = 'Safety'  # Replace with the actual user's interest
user_location = 'Old Port'  # Replace with the actual user's location
user_age = '25'  # Replace with the actual user's age
recommended_policies = recommend_policies(user_interest, user_location, user_age, df_users, df_categorised)

if not recommended_policies.empty:
    print("Recommended Policies:")
    print(recommended_policies[['Classification Number ', 'Type', 'Description ', 'Link']])
else:
    print("No recommended policies.")


# In[ ]:




