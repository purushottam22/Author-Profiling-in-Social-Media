import tweepy 
from tweepy import OAuthHandler 
from textblob import TextBlob 

def sentiment_analysis(text):
    analysis = TextBlob(text)  
    if analysis.sentiment.polarity>0: 
       return (+1)
    elif analysis.sentiment.polarity==0: 
       return 0
    else: 
       return (-1)
