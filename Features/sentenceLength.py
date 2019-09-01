import nltk
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk import sent_tokenize

def SntncLngth(text):
	sentence = sent_tokenize(text)
	tokenizer = RegexpTokenizer(r'\w+')
	num_sent=0
	avg = 0
	for sent in sentence:
		word_token = tokenizer.tokenize(sent)
		num_sent = num_sent+len(word_token)
	if(len(sentence)!=0):
		avg= float(num_sent)/len(sentence)
	#print(avg)
	return avg



