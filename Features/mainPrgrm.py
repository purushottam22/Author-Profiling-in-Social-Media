from sentenceLength import SntncLngth
from specialChar import SpclChar
from urlExtractor import UrlExtractor
from posTagging import PosTagging
from meanwordLength import MeanWrdLngth
from unrepeatedWords import UnrptdWords
from emojiCounter import EmojiCounter
from senti import sentiment_analysis
import os
import pandas as pd
#import random
import numpy as np
from sklearn.svm import SVC
import nltk
nltk.download('stopwords')


def main_functions(text1):
    A = []
    features=[]
    A.append(text1)
    features.append(SntncLngth(text1))
    features.append(SpclChar(text1))
    features.append(UrlExtractor(text1))
    features.append(MeanWrdLngth(text1))
    features.append(UnrptdWords(text1))
    d1,d2 = EmojiCounter(text1)
    features.append(d1)
    features.append(d2)
    features.append(sentiment_analysis(text1))
    return features

 
