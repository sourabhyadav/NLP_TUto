# Parts of Speech
import nltk
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer


train_text = state_union.raw("2005-GWBush.txt")
sample_text = state_union.raw("2006-GWBush.txt")
print(train_text)

# Train a pst tokenizer with our training text
cust_sent_tokenizer = PunktSentenceTokenizer(train_text)
# Apply trained tokenizer on test/sample text
tokenized = cust_sent_tokenizer.tokenize(sample_text)
print("tokenized using pst: ", tokenized)

def process_content():
    try:
        for i in tokenized:
            worrds = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(worrds)
            print(tagged)
    except Exception as e:
        print(e)

process_content()

