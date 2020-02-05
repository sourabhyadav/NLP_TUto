# Stop Words

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