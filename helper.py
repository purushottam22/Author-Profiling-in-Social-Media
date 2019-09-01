#!/usr/bin/python
#-*- coding: utf-8 -*-

import os, pickle, re
from string import punctuation

#Categories
categories_dict = {
    'algeria':1, 
    'sport':2, 
    'entertainment':3,
    'society':4, 
    'world':5, 
    'religion':6, 
}

### Tools
## Farasa Arabic NLP Toolkit
# Tokenizer
#farasaSegmenter = 'Tools/farasa/segmenter'

## Arabic StopWords List
stopWords = open("tools/list.txt").read().splitlines()

## Models directory
models = 'Models/dumps/'

## Remove Numbers and add Other punctuation
punctuation += '،؛؟”0123456789“'

class Helper():
    def __init__(self, article = False):
        self.article = article

    ##~~Pickle helpers~~#
    def getPickleContent(self, pklFile):
        with open (pklFile, 'rb') as fp:
            itemlist = pickle.load(fp)
        return itemlist

    def setPickleContent(self, fileName, itemList):
        with open(fileName+'.pkl', 'wb') as fp:
            pickle.dump(itemList, fp)
    #~~~~~~~~~~~~~~~~~~#

    #~~~ Set and get Model
    def getModel(self, name):
        model = self.getPickleContent(os.path.join(models, name+'/model_'+name+'.pkl'))
        cv = self.getPickleContent(os.path.join(models, name+'/cv_'+name+'.pkl'))
        tfidf = self.getPickleContent(os.path.join(models, name+'/tfidf_'+name+'.pkl'))
        return model, cv, tfidf

    def setModel(self, name, model, cv, tfidf):
        path = os.path.join(models, name)
        if not os.path.exists(path):
            os.mkdir(path)
        self.setPickleContent(os.path.join(models, name+'/model_'+name), model)
        self.setPickleContent(os.path.join(models, name+'/cv_'+name), cv)
        self.setPickleContent(os.path.join(models, name+'/tfidf_'+name), tfidf)
    #~~~~~~~~~~~~~~~~~~

    # Get the article content
    def getArticleContent(self, article):
        if os.path.exists(article):
            return open(article, 'r').read()

    # Drop empty lines
    def dropNline(self, article):
        if os.path.exists(article):
            content = self.getArticleContent(article)
            return re.sub(r'\n', ' ', content)
    '''
    # Get stemmed content 
    def getLemmaArticle(self, content):
        jarFarasaSegmenter = os.path.join(farasaSegmenter, 'FarasaSegmenterJar.jar')
        tmp = os.path.join(farasaSegmenter, 'tmp')
        if os.path.exists(tmp):
            os.system('rm '+tmp)
        open(tmp, 'w').write(content)
        tmpLemma = os.path.join(farasaSegmenter, 'tmpLemma')
        if os.path.exists(tmpLemma):
            os.system('rm '+tmpLemma)
        os.system('java -jar ' + jarFarasaSegmenter + ' -l true -i ' + tmp + ' -o ' + tmpLemma)
        return self.getArticleContent(tmpLemma)
    '''
    def getLemmaArticle(self, content):
        A=[]
        for txt in content:
            txt = content.split(' ')
            A.append(txt)
        print('content', content)
        print('txt', txt)
        str1=','
        tmpLemma = str1.join(txt)
        print(' temp lemma',tmpLemma)
        return txt
   



    # Remove Stop words
    def getCleanArticle(self, content):
        A=[]
        for lines in content:
            new_lines = ''
            lines1= lines.split(' ')
            for words in lines1:
                if words not in punctuation:
                    new_lines = new_lines + words+' '
            new_lines = new_lines[:-1]
            #new_line = ','.join(c for c in lines if c not in punctuation)

            #print(' content after punc', new_lines)
            new_lines2 = new_lines.split(' ')
            new_lines3=''
            for w in new_lines2:
                #print(w)
                if w in stopWords:
                    #print('w in stopword', w)
                    continue
                new_lines3 = new_lines3 + w+ ' '
            #if(new_lines3[:-1] == ' '):
            new_lines3 = new_lines3[:-1]
            #print("new line after all", new_lines3)
            #cleanWords = ''.join(if new_line not in stopWords)
            #print('content after stopwords', cleanWords)
            A.append(new_lines3)
            
        #print('OUTPUT of clean article', A)
        return A

    # Pre-processing Pipeline, before prediction (Get article Bag of Words)
    def pipeline(self, content):
        cleanArticle = self.getCleanArticle(content)
        #print(' clean article', cleanArticle)
        #lemmaContent = self.getLemmaArticle(cleanArticle)
        #print('lemma content', lemmaContent)

        #str2= ','.join(lemmaContent)
        #print(' returning string', str2)
        #cleanArticle = self.getCleanArticle(lemmaContent).split()
        #return ','.join(lemmaContent)
        return cleanArticle

    # Main function, predict content category
    def predict(self, content):
        print('content is', content)
        article = self.pipeline(content)
        model, cv, tfidf = self.getModel('sgd_94')
        vectorized = tfidf.transform(cv.transform([article]))
        predicted = model.predict(vectorized)
        keys = list(categories_dict.keys())
        values = list(categories_dict.values())
        categoryPredicted = keys[values.index(predicted[0])].upper()
        return categoryPredicted


    def get_words(self,content):
        print('content is', content)
        article = self.pipeline(content)
        return article
	



if __name__ == '__main__':
    help = Helper()
    content = 'أمرت السلطات القطرية الأسواق والمراكز التجارية في البلاد برفع وإزالة السلع الواردة من السعودية والبحرين والإمارات ومصر في الذكرى .'
    category = help.predict(content)
    print(category)
