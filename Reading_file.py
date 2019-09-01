#!/user/bin/env python
#from __future__import unicode_literal
import pandas as pd
import emoji

def read_file(file_name):
	Data_list=[]
	fopen = open(file_name, 'r')
	k=0
	for line in fopen:
		#print(line)
		if k==0:
			k+=1
			continue
		lines =line.split(',')	
		#print(lines)
		#lines1 = lines[0].split(' ')
		#print(lines1)
		papers={}
		#print(len(lines))
		#print('lines', lines)
		papers['text'] = lines[3]
		papers['Gender'] = lines[1]
		
		Data_list.append(papers)
	return Data_list

'''
def strp_text(text):
	text1= text.split(' ')
	text_list=[]
	for word in text1:
		print(' word in strip text', word)
		if word not in emoji.UNICODE_EMOJI:
			print(' not in emoji')
			text_list.append(word)
		else:
			print(' in emoji')
			A = extract_emoji(word)
			print(' A extracted emoji', A)
			for i in range(0, len(A)):
				text_list.append(A[i])
	print(' text list converted', text_list)
	return ' '.join(text_list)
	#return text_list

def strp_emoji(word):
    if word in emoji.UNICODE_EMOJI:
        return True
def extract_emoji(word):
    return ','.join(c for c in word if c in emoji.UNICODE_EMOJI)

'''
def text_has_emoji(text):
    for character in text:
        if character in emoji.UNICODE_EMOJI:
            return True
    return False

def text_separate_emoji(text):
    emoji_string = ""
    for character in text:
        if character in emoji.UNICODE_EMOJI:
            emoji_string = emoji_string + ' '+ str(character)
    return emoji_string


def full_text_separate_emoji(text):
    full_string = ""
    texts= text.split(' ')
    for words in texts:
        if text_has_emoji(words):
            for character in words:
                if character in emoji.UNICODE_EMOJI:
                    full_string = full_string + ' '+ str(character)
        else:
            full_string = full_string + ' '+words
   
    return full_string


def read_file_pandas(f_name):
	
	dt = pd.read_csv(f_name, header= 0, index_col = None)
	df = dt.values
	#print(dt.head)
	text=[]
	Age = []
	papers_list=[]
	text=df[:,4]
	Age = df[:,0]
	Gender = df[:,2]
	Lang = df[:,3]
	Id = df[:,1]
	#print(text)
	#print(Age)
	for t in range(0, len(text)):
		papers={}
		#print(' original text', text[t])
		stripped_text = full_text_separate_emoji(text[t])
		#print(type(stripped_text))
		text5 = stripped_text.split(" ")
		Pk = ''
		for i in range(len(text5)):
			if(text5[i].startswith('@') is True):
				Pk = Pk+text5[i]
			if(text5[i].startswith('#') is True):
				Pk = Pk +text5[i]
			if(text5[i].startswith('http') is True):
				Pk = Pk + text5[i]
		#stripped_text.replace(Pk,'')
		#print(Pk)
		New_Text=[]
		for words in text5:
			if words not in Pk:
				New_Text.append(words)

		New_Text = " ".join(New_Text)
		
		papers['text'] = New_Text
		papers['age1'] = Age[t]
		papers['id'] = Id[t]
		papers['gender1'] = Gender[t]
		papers['lang1'] = Lang[t]
		papers_list.append(papers)
		#print(text[t])
	return papers_list


#A= read_file_pandas()

#print(A)	
	
#print(B)

