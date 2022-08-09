# https://www.youtube.com/watch?v=D9yyt6BfgAM

import tensorflow_hub as hub
import tensorflow_text as text

preprocess_url = 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3'
encoder_url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4"

# neural entwork chain= bert_preprocess_layer , encoding_layer

bert_preprocess_model =  hub.KerasLayer(preprocess_url)

text_test=['nice movie indeed' , 'I love python programming']
text_processed = bert_preprocess_model(text_test)
print(text_processed.keys())

print(text_processed['input_word_ids'])
#for each word assigns a vlaue and is based on that.
# cls is 101
# nothing is 0
# nothing is 0

#the second layer takes in the dictionary of preprocessed text as input
bert_model = hub.KerasLayer(encoder_url)
bert_results = bert_model(text_processed)



print(bert_results.keys())

print(bert_results['pooled_output'])

# for pooled_output , for each sentence there exists a 768 dimensional vector to represent it.
# so nice-movie-indeed is represented by 768 word vector

print(bert_results['sequence_output'])
# (2 , 128  , 768)
# 2 is for the number of sentences in text_test
# each sentence consists of 128 words. If less , there is padding.
#each word is represented by a 768 dimensional vector
#for sequence_output , each word of the sentence is represented



