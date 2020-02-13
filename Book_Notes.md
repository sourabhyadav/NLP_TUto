## Important Points from the Book  

* Deep learning for natural-language processing is pattern recognition
applied to words, sentences, and paragraphs, in much the same way that computer
vision is pattern recognition applied to pixels  

* Like all other neural networks, deep-learning models don’t take as input raw text:
they only work with numeric tensors. Vectorizing text is the process of transforming text
into numeric tensors.  

This can be done in multiple ways:  
 Segment text into words, and transform each word into a vector.  
 Segment text into characters, and transform each character into a vector.  
 Extract n-grams of words or characters, and transform each n-gram into a vector.
N -grams are overlapping groups of multiple consecutive words or characters  

* the different units into which you can break down text (words, charac-
ters, or n-grams) are called tokens, and breaking text into such tokens is called tokeniza-
tion  

* All text-vectorization processes consist of applying some tokenization scheme and
then associating numeric vectors with the generated tokens.  

* There are multiple ways to
associate a vector with a token. In this section, I’ll present two major ones: one-hot
encoding of tokens, and token embedding (typically used exclusively for words, and called
word embedding)