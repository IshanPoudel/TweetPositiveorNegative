import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.utils import to_categorical

preprocess_url = 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3'
encoder_url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4"

bert_preprocess_model =  hub.KerasLayer(preprocess_url)
bert_model = hub.KerasLayer(encoder_url)

#perform one hot encoding
def get_sentence_embeddings(sentence):
    preprocessed_text = bert_preprocess_model(sentence)
    sentence_embedding = bert_model(preprocessed_text)['pooled_output']
    return sentence_embedding



#read csv file
data = pd.read_csv('tweets.csv'  , sep=";")
# print(data.head(5))

#
#group by category

# print(data.groupby('sentiment').describe())



#
# print(data['text'])

#if positive tweet classify it as 1 else classify it as 0
#change sentiment to -1 ,0 , 1
data['sentiment'] = data['sentiment'].astype('category')
data['sentiment'] = data['sentiment'].cat.rename_categories({'positive' : 2 , 'negative' : 0 , 'neutral' : 1 })

data.dropna(inplace=True)
# print(data.head(5))

X_train , X_test , y_train , y_test = train_test_split(data['text'] , data['sentiment'] , test_size=0.2 )


#create the two preprocessing layers

# print(get_sentence_embeddings(['Hey how are you?', 'Are you up for the volleyball game?']))

y_train = to_categorical(y_train, 3)
y_test = to_categorical(y_test, 3)

# #check cosine similarity
#
# embeddings = get_sentence_embeddings(['Arsenal' , 'Manchester United' , 'Melbourne Cricket Club'])
# print(cosine_similarity([embeddings[0]] , [embeddings[1]]))

#Use a functional Model for

text_input_first_layer = tf.keras.layers.Input(shape = () , dtype = tf.string , name = 'text')
preprocessed_text = bert_preprocess_model(text_input_first_layer)
sentence_embedding = bert_model(preprocessed_text)
l = tf.keras.layers.Dropout(0.1 , name = 'dropout')(sentence_embedding['pooled_output'])
l = tf.keras.layers.Dense(3 , activation = 'softmax' , name='output')(l)

model = tf.keras.Model(inputs=[text_input_first_layer] , outputs = [l])
model.summary()

model.compile(optimizer = 'adam' , loss = 'categorical_crossentropy' , metrics=['accuracy'])
model.fit(X_train , y_train , epochs=2)
model.save('tweet_sentiment_classification_multi_class.h5')
print("SAVED THE MODEL")

model.evaluate(X_test , y_test)

