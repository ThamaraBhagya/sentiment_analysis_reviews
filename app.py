import streamlit as st
import numpy as np
import pandas as pd
import re
import string
import pickle
from nltk.stem import PorterStemmer

ps = PorterStemmer()

# load model
with open('static/model/model.pickle', 'rb') as f:
    model = pickle.load(f)

# load stopwords
with open('static/model/corpora/stopwords/english', 'r') as file:
    sw = file.read().splitlines()

# load tokens
vocab = pd.read_csv('static/model/vocabulary.txt', header=None)
tokens = vocab[0].tolist()

# ----------------------------
# Preprocessing functions
# ----------------------------
def remove_punctuations(text):
    for punctuation in string.punctuation:
        text = text.replace(punctuation, '')
    return text

def preprocessing(text):
    data = pd.DataFrame([text], columns=['tweet'])
    data["tweet"] = data["tweet"].apply(lambda x: " ".join(x.lower() for x in x.split()))
    data["tweet"] = data['tweet'].apply(lambda x: " ".join(re.sub(r'^https?:\/\/.*[\r\n]*', '', x, flags=re.MULTILINE) for x in x.split()))
    data["tweet"] = data["tweet"].apply(remove_punctuations)
    data["tweet"] = data['tweet'].str.replace(r'\d+', '', regex=True)
    data["tweet"] = data["tweet"].apply(lambda x: " ".join(x for x in x.split() if x not in sw))
    data["tweet"] = data["tweet"].apply(lambda x: " ".join(ps.stem(x) for x in x.split()))
    return data["tweet"]

def vectorizer(ds):
    vectorized_lst = []
    for sentence in ds:
        sentence_lst = np.zeros(len(tokens))
        for i in range(len(tokens)):
            if tokens[i] in sentence.split():
                sentence_lst[i] = 1  
        vectorized_lst.append(sentence_lst)
    vectorized_lst_new = np.asarray(vectorized_lst, dtype=np.float32)
    return vectorized_lst_new

def get_prediction(vectorized_text):
    prediction = model.predict(vectorized_text)
    if prediction == 1:
        return 'negative'
    else:
        return 'positive'

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Sentiment Analysis", layout="centered")
st.title("💬 Sentiment Analysis App")

if "reviews" not in st.session_state:
    st.session_state.reviews = []
    st.session_state.positive = 0
    st.session_state.negative = 0

text = st.text_area("Enter your review:")

if st.button("Analyze"):
    preprocessed_txt = preprocessing(text)
    vectorized_txt = vectorizer(preprocessed_txt)
    prediction = get_prediction(vectorized_txt)

    if prediction == "negative":
        st.session_state.negative += 1
    else:
        st.session_state.positive += 1

    st.session_state.reviews.insert(0, (text, prediction))

# Show results
st.subheader("Results")
st.write(f"✅ Positive: {st.session_state.positive}")
st.write(f"❌ Negative: {st.session_state.negative}")

st.subheader("Recent Reviews")
for review, sentiment in st.session_state.reviews:
    st.write(f"- **{review}** → {sentiment}")
