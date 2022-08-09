import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model

preprocess_url = 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3'
encoder_url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4"

bert_preprocess_model =  hub.KerasLayer(preprocess_url)
bert_model = hub.KerasLayer(encoder_url)


def get_sentence_embeddings(sentence):
    preprocessed_text = bert_preprocess_model(sentence)
    sentence_embedding = bert_model(preprocessed_text)['pooled_output']
    return sentence_embedding



#read csv file
data = pd.read_csv('tweets.csv'  , sep=";")
# print(data.head(5))


#group by category

# print(data.groupby('sentiment').describe())



#
# print(data['text'])

#if positive tweet classify it as 1 else classify it as 0
#change sentiment to -1 ,0 , 1
data['positive'] = data['sentiment'].apply(lambda x:1  if x =='positive' else 0)




X_train , X_test , y_train , y_test = train_test_split(data['text'] , data['positive'] , test_size=0.2 , stratify=data['positive'])


#create the two preprocessing layers

# print(get_sentence_embeddings(['Hey how are you?', 'Are you up for the volleyball game?']))



# #check cosine similarity
#
# embeddings = get_sentence_embeddings(['Arsenal' , 'Manchester United' , 'Melbourne Cricket Club'])
# print(cosine_similarity([embeddings[0]] , [embeddings[1]]))

#Use a functional Model for

# text_input_first_layer = tf.keras.layers.Input(shape = () , dtype = tf.string , name = 'text')
# preprocessed_text = bert_preprocess_model(text_input_first_layer)
# sentence_embedding = bert_model(preprocessed_text)
# l = tf.keras.layers.Dropout(0.1 , name = 'dropout')(sentence_embedding['pooled_output'])
# l = tf.keras.layers.Dense(1 , activation = 'sigmoid' , name='output')(l)
#
# model = tf.keras.Model(inputs=[text_input_first_layer] , outputs = [l])
# model.summary()
# METRICS = [
#       tf.keras.metrics.BinaryAccuracy(name='accuracy'),
#       tf.keras.metrics.Precision(name='precision'),
#       tf.keras.metrics.Recall(name='recall')
# ]
#
# model.compile(optimizer='adam',
#               loss='binary_crossentropy',
#               metrics=METRICS)
#
# model.fit(X_train , y_train , epochs=10)
# model.save('tweet_sentiment_classification_1_or_0.h5')
# print("SAVED THE MODEL")
saved_model = load_model('tweet_sentiment_classification_1_or_0.h5' , custom_objects={'KerasLayer':hub.KerasLayer})
model = saved_model

model.evaluate(X_test , y_test)

reviews = [
    'The value is going up' ,
    'The share price of AMZN is going up' ,
    'FB are down low. The market does not look so great for them' ,
    'AMZN seem to have a very good way around their company. They are in an excellent phase to scale their production.'
    'Because gas prices are so high , TSLA seems to be doing quite badly'
]

y_predicted = model.predict(reviews)
print(y_predicted)