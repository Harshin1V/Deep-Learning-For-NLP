#  01: Deep Learning and Natural Language <br>
#  02: Cleaning Text Data<br>
#  03: Bag-of-Words Model<br>
#  04: Word Embedding Representation<br>
#  05: Learned Embedding<br>
#  06: Classifying Text<br>
#  07: Movie Review Sentiment Analysis Project<br>

# Natural Language Processing
Natural Language Processing, or NLP for short, is broadly defined as the automatic manipulation of natural language, like speech and text, by software.

The study of natural language processing has been around for more than 50 years and grew out of the field of linguistics with the rise of computers.

The problem of understanding text is not solved, and may never be, is primarily because language is messy. There are few rules. And yet we can easily understand each other most of the time. <be>

# 1. Deep Learning
Deep Learning is a subfield of machine learning concerned with algorithms inspired by the structure and function of the brain called artificial neural networks.

A property of deep learning is that the performance of these type of model improves by training them with more examples by increasing their depth or representational capacity.

In addition to scalability, another often cited benefit of deep learning models is their ability to perform automatic feature extraction from raw data, also called feature learning.<be>

# 2.Cleaning Text Data
In this lesson, you will discover how you can load and clean text data so that it is ready for modeling using both manually and with the NLTK Python library. <br>
- Text is Messy
You cannot go straight from raw text to fitting a machine learning or deep learning model.<br>
You must clean your text first, which means splitting it into words and normalizing issues such as:<br>
- Upper and lower case characters.
- Punctuation within and around words.
- Numbers such as amounts and dates.
- Spelling mistakes and regional variations.
- Unicode characters
- and much more…

# Manual Tokenization
Generally, we refer to the process of turning raw text into something we can model as “tokenization”, where we are left with a list of words or “**tokens**”.  <br>
We can manually develop Python code to clean text, and often this is a good approach given that each text dataset must be tokenized in a unique way.  <br>
For example, the snippet of code below will load a text file, split tokens by whitespace and convert each token to lowercase.
```
filename = '...' <br>
file = open(filename, 'rt')<br>
text = file.read()<br>
file.close()<br>
#split into words by white space<br>
words = text.split()<br>
#convert to lowercase<br>
words = [word.lower() for word in words]
```

# NLTK Tokenization
Many of the best practices for tokenizing raw text have been captured and made available in a Python library called the Natural Language Toolkit or NLTK for short.<be>
```
#load data<br>
filename = '...'<br>
file = open(filename, 'rt')<br>
text = file.read()<br>
file.close()<br>
#split into words<br>
from nltk.tokenize import word_tokenize<br>
tokens = word_tokenize(text) '''<be>
```

# Bag-of-Words Model
**Bag of words model** and how to encode text using this model so that you can train a model using the scikit-learn and Keras Python libraries.<br>
- The bag-of-words model is a **way of representing text data when modeling text with machine learning algorithms.**
- The approach is very simple and flexible, and can be used in **a myriad of ways for extracting features from documents.**
- A bag-of-words is a representation of text that describes the **occurrence** of words within a document.
- A vocabulary is chosen, where perhaps some infrequently used words are discarded. A given document of text is then represented using a **vector** with one position for each word in the vocabulary and a **score for each known word that appears (or not) in the document.**
- It is called a “**bag**” of words, because any information about the **order or structure of words in the document is discarded**. The model is only **concerned with whether known words occur in the document**, not where in the document.
1.  scikit-learn
2.  Keras Python Libraries
<br>

# 4.Word Embedding Representation <br>

The word embedding distributed representation and how to develop a word embedding using the Gensim Python library.
### Word Embeddings
- Word embeddings are a type of word representation that allows **words with similar meaning to have a similar representation.**
- They are a distributed representation of text that is perhaps one of the key breakthroughs for the impressive performance of deep learning methods on challenging natural language processing problems.
- Word embedding methods learn a real-valued vector representation for a predefined fixed-sized vocabulary from a corpus of text.
### Train Word Embeddings
You can train a word embedding distributed representation using the Gensim Python library for topic modeling. <br>
Gensim offers an implementation of the word2vec algorithm, developed at Google for the fast training of word embedding representations from text documents. <br>
- Embeddings are numerical representations of real-world objects that machine learning (ML) and artificial intelligence (AI) systems use to understand complex knowledge domains. They translate objects like text, images, and audio into a mathematical form that can be consumed by machine learning models and semantic search algorithms. <br>


### Use Embeddings
Once trained, the embedding can be saved to file to be used as part of another model, such as the front-end of a deep learning model. <br>

You can also plot a projection of the distributed representation of words to get an idea of how the model believes words are related. A common projection technique that you can use is the Principal Component Analysis or PCA, available in scikit-learn.<br>

The snippet below shows how to train a word embedding model and then plot a two-dimensional projection of all words in the vocabulary.<br>



```
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from matplotlib import pyplot
#define training data
sentences = [['this', 'is', 'the', 'first', 'sentence', 'for', 'word2vec'],
			['this', 'is', 'the', 'second', 'sentence'],
			['yet', 'another', 'sentence'],
			['one', 'more', 'sentence'],
			['and', 'the', 'final', 'sentence']]
#train model
model = Word2Vec(sentences, min_count=1)
#fit a 2D PCA model to the vectors
X = model[model.wv.vocab]
pca = PCA(n_components=2)
result = pca.fit_transform(X)
#create a scatter plot of the projection
pyplot.scatter(result[:, 0], result[:, 1])
words = list(model.wv.vocab)
for i, word in enumerate(words):
	pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
pyplot.show()
```
