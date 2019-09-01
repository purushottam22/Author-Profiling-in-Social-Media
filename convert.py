import pandas as pd
f_name = 
fopen = pd.read_csv(f_name)
dt = fopen.values

age = dt[:,0]
file = dt[:,1]
gen = dt[:,2]
lang = dt[:,3]
new_age = []
new_gen = []
new_lang = []
for i in range(len(age)):
	if(age[i]== 0):
		new_age.append('Below')
	elif(age[i] == 1):
		new_age.append('Between')
	elif(age[i] == 2):
		new_age.append('Above')
	elif(gen[i] == 0):
		new_gen.append('Female')
	elif(gen[i]==1):
		new_gen.append('Male')
	elif(lang[i]==0):
		new_lang.append('Iraq')
	elif(lang[i]==1):
		new_lang.append('Algeria')
	elif(lang[i]==2):
		new_lang.append('Egypt')

clean = pd.DataFrame({"File name":file,"Gender": new_gen, "Age": new_age, "Language": new_lang})
export_csv = clean.to_csv ('Predicted/Data_15/final_Truth.csv',index = None, header = True)
