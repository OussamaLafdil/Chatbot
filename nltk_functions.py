import nltk
from nltk.stem.porter import PorterStemmer
import numpy as np
import spacy
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import unidecode

#nltk.download('stopwords')
#nltk.download('punkt')

nlp = spacy.load("fr_core_news_sm")

stemmer = SnowballStemmer(language='french')

stopWords = set(stopwords.words('french'))


def tokenize(sentence):
    # Tokeniser la phrase
    doc = nlp(sentence)
    # Retourner le texte de chaque token
    return [unidecode.unidecode(X.text) for X in doc if X.text not in stopWords]


def stem(word):
    return stemmer.stem(word.lower())


def bag_of_words(tokenized_sentence, all_words):
    tokenized_sentence = [stem(w) for w in tokenized_sentence]

    bag = np.zeros(len(all_words), dtype = np.float32)
    for idx, w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1.0
    
    return bag


#b = ['organiser', 'organisateur', 'organisation', 'organisme', 'organ']
#b = [stem(w) for w in b]

#print(b)
#a = "Comment le ballon de football qui est un simple ballon si vous etes d'accord s'est-il senti après être allé à la salle de sport? Gonflé à bloc!"
#a = tokenize(a)
#print(a)