import string
import collections as ct

def SpclChar(text):
	special_chars = string.punctuation
	a=sum(v for k, v in ct.Counter(text).items() if k in special_chars)
	#print(a)
	return a
