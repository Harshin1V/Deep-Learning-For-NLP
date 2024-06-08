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
'''
filename = '...'
file = open(filename, 'rt')
text = file.read()
file.close()
# split into words by white space
words = text.split()
# convert to lowercase
words = [word.lower() for word in words] '''






