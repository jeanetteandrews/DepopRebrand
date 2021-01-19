import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#from app_store_scraper import AppStore
#from pprint import pprint
from nltk.corpus import stopwords  
from nltk.tokenize import word_tokenize 
from collections import Counter
from scipy import stats
from datetime import datetime

# scrape reviews
#depop = AppStore(country="us", app_name="depop")
#depop.review()

#pprint(depop.reviews)
#pprint(depop.reviews_count)

# create a dataframe of reviews
#df = pd.DataFrame.from_dict(depop.reviews)
#df.to_csv('extra_files/depopreviews.csv', index=False)

# create a new copy of the dataframe
copy = pd.read_csv("extra_files/depopreviews.csv")

# create lists of stopwords and punctuation
stop_words = set(stopwords.words('english'))
punc = '''!()-[]{};:’'"\, <>./?@#$%^&*_~'''


#### FIND KEYWORDS ####

# create empty list
keywords = [] 

# loop through rows of dataframe
for value in copy["review"]: 
    filtered_list = []  
    word_tokens = word_tokenize(value)
    word_tokens = [x.lower() for x in word_tokens]
    
    # filter out stopwords and punctuation
    for w in word_tokens:
        if w not in punc:  
            if w not in stop_words:  
                filtered_list.append(w) 
                
    # add keywords to empty list
    keywords.append(filtered_list)
       
# add keywords to dataframe
copy["keywords"] = keywords

# create list of all keywords
frequent_words_list = [] 

for value in copy["keywords"]: 
    for w in value:
        frequent_words_list.append(w)  
        
# find keyword frequency
counts = Counter(frequent_words_list)
labels, values = zip(*counts.items())

# sort frequency in descending order
indSort = np.argsort(values)[::-1]

# rearrange data
labels = np.array(labels)[indSort]
values = np.array(values)[indSort]

# create dataframe of keyword frequencies
keyword_df=pd.DataFrame({'keyword':labels,'frequency':values}) 
keyword_df.to_csv('extra_files/keywords.csv', index=False)

#### FIND RELEVANT UX-RELATED REVIEWS ####

# open list of relevant keywords
f = open("extra_files/relevantkeywords.txt")
lines = f.read().split()

match = []
for value in copy['keywords']:
    
    # check if keywords match in list
    for x in value:
        check = False
        if x in lines:
            check = True
    if check:
        match.append('true')
    else:
        match.append('false')

copy['match'] = match

# filter relevant reviews
relevant_df = copy.loc[copy['match'] == 'true']
relevant_df.to_csv('extra_files/relevantreviews.csv', index=False)


#### AVERAGE RATINGS OVER TIME ####

def to_datetime(d):
    return(datetime.strptime(d, '%Y-%m-%d %H:%M:%S'))

def to_timestamp(d):
    return(d.timestamp())

copy['date'] = copy['date'].apply(to_datetime)
copy['timestamp'] = copy['date'].apply(to_timestamp)

# simple linear regression
fit = stats.linregress(copy['timestamp'], copy['rating'])
copy['prediction'] = copy['timestamp']*fit.slope + fit.intercept

# plot rating over time
plt.xticks(rotation=25)
plt.plot(copy['date'], copy['rating'], 'b.', alpha=0.5)
plt.plot(copy['date'], copy['prediction'], 'r-', linewidth=3)
plt.title("Average Ratings Over Time")
plt.xlabel("date")
plt.ylabel("rating")
plt.savefig("extra_files/plot.jpeg")
