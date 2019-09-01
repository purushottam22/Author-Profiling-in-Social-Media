import nltk
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk import sent_tokenize
import math

def UnrptdWords(text):
	tokenizer = RegexpTokenizer(r'\w+')
	words = tokenizer.tokenize(text)
	seen=set()
	dups=set()
	for word in words:
	  if word in seen:
	    if word not in dups:
	      dups.add(word)
	  else:
	    seen.add(word)
	unrep=set()
	unrep=seen.difference(dups)
	#print(unrep)
	#print(seen)
	#print(dups)
	b=len(unrep)
	return b

