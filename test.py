from nltk_functions import tokenize,stem,bag_of_words
import spacy
import nltk

import nltk
from nltk.corpus import stopwords

from nltk.stem.snowball import SnowballStemmer
stemmer1 = SnowballStemmer(language='french')

#nltk.download('stopwords')
from nltk.corpus import stopwords
stopWords = set(stopwords.words('french'))




nlp = spacy.load("fr_core_news_sm")

def return_token(sentence):
    # Tokeniser la phrase
    doc = nlp(sentence)
    # Retourner le texte de chaque token
    return [X.text for X in doc if X.text not in stopWords]

def stem1(word):
    return stemmer1.stem(word.lower())

b = ['organiser', 'organisateur', 'organisation', 'organisme', 'organ']
b = [stem(w) for w in b]
c = [stem1(w) for w in b]
print(b)
print(c)
a = "Comment le ballon de football qui est un simple ballon si vous etes d'accord s'est-il senti après être allé à la salle de sport? Gonflé à bloc!"
#a = return_token(a)
#print(a)