import nltk
#nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

ex_sent = "This is a simple sentence, showing the word filteration."
stop_words = set(stopwords.words('english'))
words_token = word_tokenize(ex_sent)
total_words=len(words_token)
filterd_sentence= [w for w in words_token if not w in stop_words]
total_stopwords=len(filterd_sentence)
avg=float(total_words)/total_stopwords
print(avg)



