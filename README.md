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

# 3.Bag-of-Words Model
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
# Use .key_to_index to get a dictionary of words and their indices
X = model.wv[model.wv.key_to_index]
pca = PCA(n_components=2)
result = pca.fit_transform(X)
#create a scatter plot of the projection
pyplot.scatter(result[:, 0], result[:, 1])
# Use .index_to_key to get a list of words in the vocabulary
words = list(model.wv.index_to_key)
for i, word in enumerate(words):
	pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
pyplot.show()
```

# 5:Learned Embedding <br>
Discover how to learn a word embedding distributed representation for words as part of fitting a deep learning model

## Embedding Layer
- **Keras** offers an **Embedding layer** that can be used for n**eural network**s on **text data.**
- It requires that the **input data be integer encoded so that each word is represented by a unique integer**. This data preparation step can be **performed** using the **Tokenizer** **API** also provided with Keras.
- The Embedding layer is initialized with **random weights** and will **learn an embedding for all of the words in the training dataset**. You must specify the input_dim which is the size of the vocabulary, the output_dim which is the size of the vector space of the embedding, and optionally the input_length which is the number of words in input sequences.
```
layer = Embedding(input_dim, output_dim, input_length=??)
```
Or, more concretely, a vocabulary of 200 words, a distributed representation of 32 dimensions and an input length of 50 words.
```
layer = Embedding(200, 32, input_length=50)
```
## Embedding with Model
<br>
- The Embedding layer can be used as the front-end of a deep learning model to provide a rich distributed representation of words, and importantly this representation can be learned as part of training the deep learning model.
- For example, the snippet below will define and compile a neural network with an embedding input layer and a dense output layer for a document classification problem.
- When the model is trained on examples of padded documents and their associated output label both the network weights and the distributed representation will be tuned to the specific data.
<br>

```
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
#define problem
vocab_size = 100
max_length = 32
#define the model
model = Sequential()
model.add(Embedding(vocab_size, 8, input_length=max_length))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
#compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#summarize the model
print(model.summary())
```
- It is also possible to initialize the Embedding layer with pre-trained weights, such as those prepared by Gensim and to configure the layer to not be trainable. This approach can be useful if a very large corpus of text is available to pre-train the word embedding.
<br>

# 6:Classifying Text
- Discover the standard deep learning model for classifying text used on problems such as sentiment analysis of text.

## Document Classification
- Text classification describes a general class of problems such as predicting the sentiment of tweets and movie reviews, as well as classifying email as spam or not.
- It is an important area of natural language processing and a great place to get started using deep learning techniques on text data.
- Deep learning methods are proving very good at text classification, achieving state-of-the-art results on a suite of standard academic benchmark problems.

## Embeddings+CNN
- The modus operandi for text classification involves the use of a word embedding for representing words and a Convolutional Neural Network or CNN for learning how to discriminate documents on classification problems.
### The architecture is comprised of three key pieces:
1. Word Embedding Model: A distributed representation of words where different words that have a similar meaning (based on their usage) also have a similar representation.
2. Convolutional Model: A feature extraction model that learns to extract salient features from documents represented using a word embedding.
3. Fully-Connected Model: The interpretation of extracted features in terms of a predictive output.
- This type of model can be defined in the Keras Python deep learning library. The snippet below shows an example of a deep learning model for classifying text documents as one of two classes.

```
# define problem
vocab_size = 100
max_length = 200
# define model
model = Sequential()
model.add(Embedding(vocab_size, 100, input_length=max_length))
model.add(Conv1D(filters=32, kernel_size=8, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
print(model.summary())
```
# 7:Movie Review Sentiment Analysis Project
- Discover how to prepare text data, develop and evaluate a deep learning model to predict the sentiment of movie reviews.
- I want you to tie together everything you have learned in this crash course and work through a real-world problem end-to-end.

## Movie Review Dataset
- The Movie Review Dataset is a collection of movie reviews retrieved from the imdb.com website in the early 2000s by Bo Pang and Lillian Lee. The reviews were collected and made available as part of their research on natural language processing.
- Movie Review Polarity Dataset (review_polarity.tar.gz, 3MB)
From this dataset you will develop a sentiment analysis deep learning model to predict whether a given movie review is positive or negative.

## Develop and evaluate a deep learning model on the movie review dataset:

- Download and inspect the dataset.
- Clean and tokenize the text and save the results to a new file.
- Split the clean data into train and test datasets.
- Develop an Embedding + CNN model on the training dataset.
- Evaluate the model on the test dataset.


