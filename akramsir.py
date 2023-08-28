import nltk
import os

# Assuming you have the necessary file(s) in your working directory

# List the files in the working directory
files_in_directory = os.listdir()

# Choose the file you want to read (for now, let's assume it's the first file in the list)
chosen_file = files_in_directory[0]

# Load text into an object called "text"
with open(chosen_file, 'r', encoding='utf8', errors='ignore') as f:
    text = f.read()

print('Read file "{name}" with length {length} characters'.format(name=chosen_file, length=len(text)))

# 1.(b) Tokenize text into sentences HERE:
nltk.download('punkt')

def tokenize_sentences(text):
    sentences = text

    return nltk.sent_tokenize(text)

sentences = tokenize_sentences(text)

# 1.(c) Clean and Preprocess documents (sentences) for our matrix
import string

translation = str.maketrans('', '', string.punctuation)

def preprocess_text(sentences):
    # Clean text HERE or in the parameters of BOW or TFIDF:
    #https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
    #https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html

    cleaned = [w.lower() for w in sentences]
    cleaned = [w.translate(translation) for w in cleaned]

    return cleaned

cleaned = preprocess_text(sentences)

# create TFIDF (or BOW) matrix on SENTENCES
import pandas as pd

# build BOW matrix HERE (from your cleaned sentences)
# remember we need both a model ("CountVec" in the code examples)
# and a matrix ("X" in the code examples)
from sklearn.feature_extraction.text import CountVectorizer


# OR

# build TFIDF matrix HERE (from your cleaned sentences)
# remember we need both a model ("TfidfVec" in the code examples)
# and a matrix ("X" in the code examples)
from sklearn.feature_extraction.text import TfidfVectorizer

TfidfVec = TfidfVectorizer(stop_words='english') #see documentation for options!
tfidf = TfidfVec.fit_transform(cleaned)

X = pd.DataFrame(tfidf.toarray(), columns = TfidfVec.get_feature_names_out(), dtype='float32')
print(X.head())

# build the bot reponse:
def respond(user_input):
    bot_response = ''

    # transform user query with our model HERE
    query = TfidfVec.transform([user_input])

    # get dot-product of our query-vector and matrix HERE
    cosine_sim = query.dot(X.T)

    # get index of maximum similarity HERE
    max_sim_index = cosine_sim.argmax()

    #if there's nothing like the user query in our matrix, give a standard response
    if max_sim_index == 0:
        return 'Samajh nahi aya, theek se pucho?'

    # fetch the sentce from our (original) sentence vector by max_sim_index and return it
    bot_response = sentences[max_sim_index]

    return bot_response

# START THE BOT
#prompt a dialog with the user
os.system('cls||clear')

print("""

        **************************************************
        |--------------Akram Sir ka ChatBot--------------|
        **************************************************

""")

print("Akram Sir: Good morning, Kya naam hai tumhara?")

#while the dialog is ongoing...
flag = True
while(flag == True):
    # transform user input
    user_input = input("Kunal: ")
    user_input = user_input.lower()

    # cue to end the conversation
    if(user_input == 'bye sir'):
        flag=False
        print("Akram Sir: OK, chalo bye, dhyaan rakho aur class aaya karo!")
        exit()
    # otherwise, respond!

    if(user_input == 'kunal'):
        print("Akram Sir: Class me toh kabhi dekhe nahi tmko. Kuch puchna hai kya tumko?")
    else:
        print("Akram Sir: ", end="")
        print(respond(user_input))