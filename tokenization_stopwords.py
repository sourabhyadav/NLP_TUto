# Tokenizing

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

sample_sentence = "Hello Mr. Sourbh, How are you doing today? Also do you like NLP-tuto! Tell me how is it? Sky is pinkish-fblue"

print("Sentence tokenize:",sent_tokenize(sample_sentence))
print("Word tokenize:", word_tokenize(sample_sentence))

"""
for word in word_tokenize(sample_sentence):
    print(word)

for sent in sent_tokenize(sample_sentence):
    print(sent)
"""

# StopWOrds

from nltk.corpus import stopwords
stop_words = set(stopwords.words("english")) # Creating a stop word of english
#print("English Stop words:", stop_words)

words = word_tokenize(sample_sentence)

filtered_sent = []

for w in words:
    if w not in stop_words:
        filtered_sent.append(w)

print("filtered sent: ", filtered_sent)

# 1-liner for loop
fil_sent = [w for w in words if w not in stop_words]
print("Pythonic form: ", fil_sent)