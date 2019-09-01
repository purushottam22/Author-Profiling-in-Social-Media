import pandas as pd

fopen1_age = pd.read_csv('/home/purushottam/Desktop/IIT_patna/FIRE/New/For_send (copy)/Predicted/Data_19/Predicted_Age_19.csv',header= 0, index_col = None )
print("readed file 1")
fopen2_gen = pd.read_csv('/home/purushottam/Desktop/IIT_patna/FIRE/New/For_send (copy)/Predicted/Data_19/Predicted_Gender_19.csv',header= 0, index_col = None )
print("readed file 2")
fopen3_lang = pd.read_csv('/home/purushottam/Desktop/IIT_patna/FIRE/New/For_send (copy)/Predicted/Data_19/Predicted_Lang_19.csv',header= 0, index_col = None )
print("readed file 3")

df1 = fopen1_age.values
df2 = fopen2_gen.values
df3 = fopen3_lang.values

id_1 = df1[:,1]
id_2 = df2[:,1]
id_3 = df2[:,1]

true_age = df1[:,0]
true_gen = df2[:,0]
true_lang = df3[:,0]

file_name = id_1
Id_set = set(file_name)
Id_list = list(Id_set)
len_Id = len(Id_set)
Final_list_new = []
Final_list=[]
preds_age = []
preds_gen = []
preds_lang = []
preds_id = []
print("loop starts now")
for  j in range(0, len_Id):
    id = Id_list[j]
    papers={}
    papers['key']= id
    
    for k in range(0, len(file_name)):
        if id_2[k] == id:
            preds_gen.append(true_age[k])
        if id_3[k] == id:
            preds_lang.append(true_lang[k])
        if id_1[k] == id:
            preds_age.append(true_age[k])
		    preds_id.append(id)

    papers['Age'] = preds_age
    papers['Gender'] = preds_gen
    papers['Lang'] = preds_lang
    papers['Id'] = preds_id
    Final_list.append(papers)
print("loop has been ended")
Final_list_new.append(Final_list)
print(len(preds_id))
print(len(preds_age))
print(len(preds_gen))
print(len(preds_lang))

clean = pd.DataFrame({"File name":preds_id,"Gender": preds_gen, "Age": preds_age, "Language": preds_lang})
export_csv = clean.to_csv ('Predicted/Data_19/Predicted_Truth.csv',index = None, header = True)
