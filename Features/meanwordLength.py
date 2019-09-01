import nltk
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk import sent_tokenize

def MeanWrdLngth(text):
	tokenizer=RegexpTokenizer(r'\w+')
	word_tokens=tokenizer.tokenize(text)
	word_tokens_a=text.split(' ')
	num_words=len(word_tokens)
	print(num_words)
	num_chars=0
	a=0
	for word in word_tokens_a:
		num_chars=num_chars+(len(word))
	if(num_words!=0):
		a = float(num_chars)/num_words
	#print(a)
	return a
