# Tokenizing

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

sample_sentence = "Hello Mr. Sourbh, How are you doing today? Also do you like NLP-tuto! Tell me how is it? Sky is pinkish-fblue"

print("Sentence tokenize:",sent_tokenize(sample_sentence))
print("Word tokenize:", word_tokenize(sample_sentence))


for word in word_tokenize(sample_sentence):
    print(word)

for sent in sent_tokenize(sample_sentence):
    print(sent)

