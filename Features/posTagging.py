import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk import sent_tokenize

def PosTagging(text):
	stop_words = set(stopwords.words('english'))
	tokenized = sent_tokenize(text) 
	for i in tokenized: 
	      
	    # Word tokenizers is used to find the words  
	    # and punctuation in a string 
	    wordsList = nltk.word_tokenize(i) 
	  
	    # removing stop words from wordList 
	    wordsList = [w for w in wordsList if not w in stop_words]  
	  
	    #  Using a Tagger. Which is part-of-speech  
	    # tagger or POS-tagger.  
	    tagged = nltk.pos_tag(wordsList) 
	  
	    #print(tagged)
	    return tagged 


