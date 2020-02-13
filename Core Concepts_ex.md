## Few NLP Terms

**Tokenizing?**
* Tokenizing is Grouping. When we group something with some constraints is called tokenizing.
* There are 2 basic types of tokenizer:
    * Word Tokenizer
    * Sentence Tokenizer

 **Corpora:**
 * Body of the given text e.g. presidential speeches, medical documents etc.

 **Lexicons:**
 * Words and their meanings.
 * Basically its not just english meaning but here meaning changes based on the context. Like "bull" is investment terms and also animal in different contexts.

**Stop Words:**
* Words which people use sarcastically can be a stop-word, some words which you don't care about them.
* They are fillers in the sentence. Basically you wanna remove them.
* Sample stop words for english corpora are is, and, whom your etc.
* Actually these words make a lot of sense to us but for Data Science it becomes different to use
* Basically they even if you remove these words the meaning of sentence remains mostly same

**Stemming:**
* Finding the root of the word is called steming. E.g. reading -> read, ridden -> ride etc.
* This becomes necessary because you in english we can use many variant of words in the different sentences with same meaning.

**Parts of Speech:**
* This collects different parts of speech in a given paragraph like nouns, verbs, adjectives etc.

**Chunking:** [TODO]

**Chinking:** 
* Excluding something from existing chunks is called chinking

**Named Entity:**
* It is nothing but chunking the parts of speech with some sort of named entitity. It is useful to chunk and understand the sentence better.
* Few examples of named entity is : NAME, LOCATION, TIME, DATE, MONEY, PERCENT etc.

**Lemmatizing:**
* It is similar to stemming but here the word would be replaced by a synonyms  

**Word Embedding:**  
* **ONe-hot encoding:** This kind of word embedding is simple hardcoded and which is at very high-dimensional. Moreover, it is very sparse.
*  **Learn word embeddings:** This kind of word embedding is trainable i,e, it can be learned from given data/corpus. This makes them dense and low-dimensional.  

