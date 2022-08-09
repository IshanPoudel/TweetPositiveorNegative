from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax

#Based on pretrained twitter models

MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)


#Add preprocessing_text
def preprocess(text):
   new_text = []
   for t in text.split(" "):
       t = '@user' if t.startswith('@') and len(t) > 1 else t
       t = 'http' if t.startswith('http') else t
       new_text.append(t)
   return " ".join(new_text)


def polarity_scores_roberta(example):
   example = preprocess(example)
   encoded_text = tokenizer(example, return_tensors='pt')
   output = model(**encoded_text)
   scores = output[0][0].detach().numpy()
   scores = softmax(scores)
   scores_dict = {

       'roberta_neg': scores[0],
       'roberta_neu': scores[1],
       'roberta_pos': scores[2]
   }

   return scores_dict