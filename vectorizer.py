import fastText
import nltk
import numpy as np
from weightings import weight

model = fastText.FastText.load_model("../model.bin")
# Replace this location with your fastText pretraied model
# https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md

def convert_word(word):
	word, tag = word

	score = weight(tag)
	# print(tag, score)
	raw = np.array(model.get_word_vector(word))
	raw[2] = 1.0
	# final = raw * score
	final = np.multiply(raw, score)

	return final

def get_sentence_vector(sentence):
	tokens = nltk.word_tokenize(sentence)
	tokens = nltk.pos_tag(tokens)

	words = np.array(map(convert_word, tokens))

	return np.average(a=words, axis=0)

