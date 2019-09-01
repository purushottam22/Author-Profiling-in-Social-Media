from aravec1 import get_word_vec
from helper import Helper
import gensim
from keras.preprocessing.text import Tokenizer
import numpy as np
from Reading_file_train import read_file_pandas_train
from Reading_file_test import read_file_pandas_test
t_model = gensim.models.Word2Vec.load('/home/purushottam/Desktop/IIT_patna/FIRE/New/For_send/Word_vec_Model/full_uni_cbow_300_twitter.mdl')
from Model import Train_model
import random
import itertools
from Model_Age import word_embed_meta_data
from Model_Age import transforming_test_unseen
import numpy as np
from keras.datasets import imdb
from keras.models import Sequential, Model
from keras.layers import Dense, Input, LSTM, Merge
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from sklearn import preprocessing 
from sklearn.preprocessing import OneHotEncoder
from keras.utils import np_utils
import pandas as pd
from keras.callbacks import EarlyStopping
from keras.models import model_from_json
from mainPrgrm import main_functions
from keras.layers.merge import Concatenate
from keras.layers.normalization import BatchNormalization

def Reading_data(Data):
    
    class_x = []
    documents_x=[]
    Id = []
    gen = []
    age = []
    lang = []
    features=[]
    for dt in Data:
        dt_text = dt['text']
        features.append(main_functions(dt_text))
        gen.append(int(dt['gender1']))
        lang.append(int(dt['lang1']))
        age.append(int(dt['age1']))
        documents_x.append(dt['text'])
        Id.append(dt['id'])
        #print('dt ', dt_text)
    return documents_x, Id, gen, age, lang, features



if __name__ == '__main__':
    Test_file  = '/home/purushottam/Desktop/IIT_patna/FIRE/Test_A.P./Data_15.csv'
    #Train_file = '/home/purushottam/Desktop/IIT_patna/FIRE/Author_profiling/Data_27 (copy).csv'
    Train_file ='/home/purushottam/Desktop/IIT_patna/FIRE/Author_profiling/Data_27.csv'
    Test_data = read_file_pandas_test(Test_file)
    Data = read_file_pandas_train(Train_file)
    Train_content, Train_Id,  Train_gen, Train_age, Train_lang, Train_features = Reading_data(Data)
	
    Test_content =[]
    Test_features = []
    Test_Id=[]
    for dt in Test_data:
        Test_content.append(dt['text'])
        Test_features.append(main_functions(dt['text']))
        Test_Id.append(dt['id'])
    '''
    print("lenght of Train id:", len(Train_Id),"length iof documt.", len(Train_content), "length of content:", len(Train_gen))
    

    max_tweet_length = 10
    help = Helper()

    words =  help.get_words(Train_content)
    #print('words', words)
    tokenizer, embedding_matrix,vocab = word_embed_meta_data(words, 300)
    x1_test_converted = transforming_test_unseen(vocab, Test_content)
    
    train_sequences = tokenizer.texts_to_sequences(Train_content)
    test_sequences = tokenizer.texts_to_sequences(x1_test_converted)
    X1_train = sequence.pad_sequences(train_sequences, maxlen=max_tweet_length)

    Train_features = sequence.pad_sequences(Train_features, maxlen=max_tweet_length)
    print("type of train features:", Train_features)
    print("type of x train", X1_train)
    #print("train features:", Train_features)
    #X_train = list(X_train)
    Test_features = sequence.pad_sequences(Test_features, maxlen=max_tweet_length)
    '''
    le = preprocessing.LabelEncoder()
    le1 = preprocessing.LabelEncoder()
    le2 = preprocessing.LabelEncoder()
    '''
    le.fit(y1_train)
    Y_train11 = le.transform(y1_train)
    Y_train = np_utils.to_categorical(Y_train11)
    '''

    le.fit(Train_age)
    le1.fit(Train_gen)
    le2.fit(Train_lang)

    age_train2 = le.transform(Train_age)
    gen_train2 = le1.transform(Train_gen)
    lang_train2 = le2.transform(Train_lang)

    age_train3 = np_utils.to_categorical(age_train2)
    gen_train3 = np_utils.to_categorical(gen_train2)
    lang_train3 = np_utils.to_categorical(lang_train2)

    #print('x train', X1_train)
    print("Train features", Train_features)
    #nb_words = len(tokenizer.word_index) + 1
    #embedding_vecor_length = 300
    
    '''
    embedding_layer = Embedding(nb_words, embedding_vecor_length, weights=[embedding_matrix],  input_length=max_tweet_length)
    lstm_layer =LSTM(10)
    
    sequence_1_input = Input(shape=(max_tweet_length,), dtype='int32')
    embedded_sequences_1 = embedding_layer(sequence_1_input)
    x11 = lstm_layer(embedded_sequences_1)

    auxiliary_input1 = Input(shape=(10,), dtype= 'float32', name='aux_input1')
    x1 = concatenate([x11, auxiliary_input1])
    merged = BatchNormalization()(x1)
    merged = Dense(10, activation= 'relu')(merged)
    #merged = BatchNormalization()(merged)
    preds = Dense(3, activation='sigmoid')(merged)
    model = Model(inputs=[sequence_1_input, auxiliary_input1], outputs=preds)
    print(model.summary())
    '''

    '''
    sequence_1_input = Input(shape=(max_tweet_length,), dtype='int32')
    embedded_sequences_1 = embedding_layer(sequence_1_input)
    x11 = lstm_layer(embedded_sequences_1)

    auxiliary_input1 = Input(shape=(10,), dtype= 'float32', name='aux_input1')
    x12 = Dense(10, activation= 'relu')(auxiliary_input1)

    concatenated = concatenate([x11, x12])
    preds = Dense(3, activation='sigmoid')(concatenated)
    model = Model(inputs=[sequence_1_input, auxiliary_input1], outputs=preds)
    '''
    #model.fit(X_train, y_train, validation_split=0.33, epochs=5, batch_size = 256)

    
    model = Sequential()
    model.add(Dense(10, input_dim = 8))
    #model.add(Dense(15, activation='relu'))
    model.add(Dense(8, activation='relu'))
    #model.add(Dense(5, activation='relu'))
    model.add(Dense(3, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    #model.fit(X_train, y_train, validation_split=0.33, epochs=5, batch_size = 256)'
    
    #model.fit(X_train, y_train, validation_split=0.33, epochs=5, batch_size = 256)
    print("for Age")
    earlyStop=EarlyStopping(monitor="val_loss",verbose=2,mode='min',patience=3)
    model.fit( Train_features, age_train3, validation_split= 0.2, epochs=20, batch_size = 256, callbacks=[earlyStop])
    '''
    print(X1_train)
    print(Train_features)
    print(age_train3)
    print("for Age")
    #earlyStop=EarlyStopping(monitor="val_loss",verbose=2,mode='min',patience=3)
    #model.fit(Input = [np.array(X1_train), np.array(Train_features)] , output = np.array(age_train3),  validation_split= 0.2, epochs=20, batch_size = 256, callbacks=[earlyStop])

    # serialize model to JSON
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model.h5")
    print("Saved model to disk")

    # later...

    # load json and create model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    print("Loaded model from disk")

    # evaluate loaded model on test data
    loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    #score = loaded_model.evaluate(X, Y, verbose=0)
    #print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
    preds = loaded_model.predict([X1_test, Test_features])
    print(preds)
    Id_test_list = []
    Id_test_list.append(Test_Id)

    preds_list=[]
    preds_list.append(preds)
    labels_list=[]
    labels=[]
    print("length of preds:", len(preds_list))
    for i in range(len(preds_list)):
        for j in range(0, len(preds_list[i])):
            max_value = np.max(preds_list[i][j])
            max_inden = np.where(preds_list[i][j] == max_value)
            #print("type of max_inden:", type(max_inden))
            max_inden = list(itertools.chain(*max_inden)) 
            labels.append(max_inden)
        #labels = (np.array(labels))

    labels1=[]
    for i in labels:
        labels1.append(i[0])

    labels = []
    labels.append(labels1)
    
    clean = pd.DataFrame({"File name":Test_Id,"Age":labels1})
    export_csv = clean.to_csv ('All_chart_age',index = None, header = True)

    Final_list_new=[]
    for i in range(0, len(Id_test_list)):
        Id_test = Id_test_list[i]
        Id_set = set(Id_test)
        Id_list = list(Id_set)
        len_Id = len(Id_set)
        Final_list=[]
        for  j in range(0, len_Id):
            id = Id_list[j]
            papers={}
            papers['key']= id
            preds_list_new=[]
            for k in range(0, len(Id_test)):
                if Id_test[k] == id:
                    ab = labels[i][k]

                    preds_list_new.append(ab)
            papers['preds'] = preds_list_new
            Final_list.append(papers)
        Final_list_new.append(Final_list)

    #print("length of labels:", len(labels_list))
    #print(' final list', Final_list_new)

    i =0
    file_name_list = []
    for final_list1 in Final_list_new:
        i = i+1
        file_name = 'predicted_age' + str(i) +'.csv'
        #fopen = open(file_name, 'w')
        final_label =[]
        id_list = []
        for item in final_list1:
            id = item['key']
            labels1 = item['preds']
            num_1 = labels1.count(1)
            num_0 = labels1.count(0)
            num_2 = labels1.count(2)
            max1 = max(num_0, num_1, num_2)
            if (max1 == num_0):
                final_label.append(0)
            elif(max1 == num_1):
                final_label.append(1)
            elif(max1 == num_2):
                final_label.append(2)
            id_list.append(id)

        clean = pd.DataFrame({"File name":id_list,"Age":final_label})
        export_csv = clean.to_csv (file_name,index = None, header = True)

    #np.savetxt('results.csv', labels,  delimiter='\n')
    
    D1 = 0
    D2 = 0
    #D3 = 0
    P = []
    for i in range(len(labels)):

        if(i!= 0):

            if(Id[i] != Id[i-1] ):
                z = max(D1, D2)
                if(z == D1):
                    P.append(0)
                elif(z == D2):
                    P.append(1)
                #elif(z == D3):
                    #P.append(2)

                D1 = 0
                D2 = 0
                #D3 = 0

        if(labels[i] == 0):
            D1+=1
        if(labels[i] == 1):
            D2+=1
        #if(labels[i] == 2):
            #D3+=1

        if(i == len(labels)-1):
            z = max(D1, D2)
            if(z == D1):
                P.append(0)
            elif(z == D2):
                P.append(1)
            #elif(z == D3):
                #P.append(2)

                D1 = 0
                D2 = 0
                #D3 = 0

    print("length of P:", len(P))
    
    #np.savetxt('final_result.csv', P,  delimiter='\n')
    # Final evaluation of the model
    #scores = model.evaluate(X_test, y_test, verbose=0, batch_size = 256)
    #print("Accuracy: %.2f%%" % (scores[1]*100))
	
	
    '''
