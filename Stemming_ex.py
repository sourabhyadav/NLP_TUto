
# Steming

from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

ps = PorterStemmer()

example_words = ["pythoning", "pythoned", "pythonly", "unpython"]

stemed_words = [ps.stem(w) for w in example_words]
print("Stemed Words: ", stemed_words)

sent_1 = "It is very important to write code in pythonic way. Once the code is written in pythonic way you can call yourself as pythoned code-styled"
token_words = word_tokenize(sent_1)
stemed_words_1 = [ps.stem(w) for w in token_words]
print("New stemped words", stemed_words_1)