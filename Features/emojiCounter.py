import emoji
import regex

def EmojiCounter(text):
		emoji_counter = 0
		data = regex.findall(r'\X', text)
		for word in data:
		     if any(char in emoji.UNICODE_EMOJI for char in word):
			     emoji_counter += 1
			     #Remove from the given text the emojis
			     text = text.replace(word, '') 
		words_counter = len(text.split())

		#print(emoji_counter)
		#print(words_counter)
		return emoji_counter, words_counter
