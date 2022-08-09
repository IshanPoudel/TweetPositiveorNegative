import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')

import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('vader_lexicon')

from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm.notebook import tqdm

data = pd.read_csv('tweets.csv' , sep=';')

#Rename categories
data['sentiment'] = data['sentiment'].astype('category')
data['sentiment'] = data['sentiment'].cat.rename_categories({'positive' : 2 , 'negative' : 0 , 'neutral' : 1 })

sia = SentimentIntensityAnalyzer()

#Add preprocessing_text
def preprocess(text):
   new_text = []
   for t in text.split(" "):
       t = '@user' if t.startswith('@') and len(t) > 1 else t
       t = 'http' if t.startswith('http') else t
       new_text.append(t)
   return " ".join(new_text)







def get_polarity_score(text):
   text = preprocess(text)
   return sia.polarity_scores(text)