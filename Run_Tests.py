
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from Vader_Model import get_polarity_score
from Roberta_Model import polarity_scores_roberta

# saved_model = load_model('tweet_sentiment_classification_1_or_0.h5' , custom_objects={'KerasLayer':hub.KerasLayer})
x_test = ['I  love the way SpaceX does its testing .' ]

# y_predicted = saved_model.predict(x_test)
# print(y_predicted.flatten())

print(get_polarity_score(x_test[0]))
print(polarity_scores_roberta(x_test[0]))
