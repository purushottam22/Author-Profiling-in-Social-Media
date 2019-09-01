import nltk
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk import sent_tokenize

def UrlExtractor(text):
	if (text.find('https:') != -1): 
	  return 1
	else: 
	  return 0

