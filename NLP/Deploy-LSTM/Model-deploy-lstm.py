## Import libs

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.models import  Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense,LSTM ,Dropout, Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences

import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import pandas as pd 
import numpy as np
import seaborn as sb 
import matplotlib.pyplot as plt 
import pickle
import time
import emoji
import base64


@st.cache(allow_output_mutation=True)
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()
def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = '''
    <style>
    body {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)
    return
set_png_as_page_bg('deathnote.jpg')


## Read file
def main():
    st.markdown("<h1 style='text-align: center; color: red;'>Sentence Generation using LSTM</h1>", unsafe_allow_html=True)
    #st.markdown("<h2 style='text-align: center; color: blue;'>*--Built using LSTM--*</h2>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: white;'>*This a Deep Learning model and is trained on Deathnote anime storyline , hence please provide the input from the storyline*</h3>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: yellow;'>*Enter the number of words to be predicted*</h3>", unsafe_allow_html=True)
    
    f= open('Deathnote-script.txt' , encoding = 'utf-8')
    d= " "
    x = []
    for i in f:
        x.append(i)
    xx = d.join(x)
    data = xx.split("\n")
    final_data = " ".join(data)

    ## Clean text 
    @st.cache(allow_output_mutation=True)
    def clean_text(doc):
        tokens = doc.split(" ")   #white space sep
        punc = str.maketrans("","",string.punctuation) # all punc
        tokens = [w.translate(punc) for w in tokens]  #remove punc
        tokens = [word for word in tokens if word.isalpha()]  #only alpha
        tokens = [word.lower() for word in tokens]  #lower
        return tokens

    tokens = clean_text(final_data)
    token_length = len(tokens)
    input_length = 50+1
    lines = []

    for i in range(input_length , len(tokens)):
        seq = tokens[i-input_length : i]  #0 to inp length
        line = " ".join(seq)   # join to make inp sequence
        lines.append(line)    # append in list
        
    ## Tokenize
   
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)   #tokens applied on lines
    seq = tokenizer.texts_to_sequences(lines)
    seq = np.array(seq)
    voc_size = len(tokenizer.word_index) + 1 # as it was started from 0
    seq_length = 50

    ## Model making
    
    model = Sequential()
    model.add(Embedding(voc_size , 50 , input_length = seq_length))
    model.add(LSTM(100 , return_sequences = True))
    model.add(Dropout(0.4))
    model.add(LSTM(80))
    model.add(Dense(100 , activation= 'relu'))
    model.add(Dense(voc_size , activation = 'softmax'))


    ## Load model using checkpoint
    checkpoint_path = "lstm-model-DN-v1.ckpt"
    model.load_weights(checkpoint_path)


    ## predict func
    def generate_text(model , tokenizer , text_seq_length , seed_text , n_words):
        final_ans = []
        for n  in range(n_words):
            encoded = tokenizer.texts_to_sequences([seed_text])[0]
            encoded = pad_sequences([encoded] , maxlen = text_seq_length , truncating = 'pre')
            y_pred = np.argmax(model.predict(encoded), axis=-1)
            #y_pred = model.predict(encoded)
            
            pred_word =  " "
            
            for word , index in tokenizer.word_index.items():
                if index == y_pred:
                    pred_word = word
                    break
            seed_text = seed_text + " " + pred_word
            final_ans.append(pred_word)
        return " ".join(final_ans)

    
    num = st.selectbox("" , [0 , 10,20,30,40,50,60,70,80,90,100]) 
    if num != 0 :
        st.markdown("<h3 style='text-align: center; color: Red;'>Enter the index for which you would like to predict :</h3>", unsafe_allow_html=True)
        ind = st.slider("  " , 0,9000)
        if ind != 0 :
            inp = lines[ind]
            st.markdown("<h2 style='text-align: center; color: White;background :rgba(66, 240, 50, 0.6)'>Input:</h2>", unsafe_allow_html=True)
            st.markdown(f""" <h3 style='text-align: center; color: White;background :rgba(53, 184, 240, 0.9);'>{inp}</h3>""", unsafe_allow_html=True)
            st.markdown("<h2 style='text-align: center; color: White;background :rgba(66, 240, 50, 0.6)'>Predicted Output:</h2>", unsafe_allow_html=True)
            st.markdown(f"""<h3 style='text-align: center; color: white;background :rgba(53, 184, 240, 0.9);'> {generate_text(model , tokenizer , seq_length , inp , num)}</h3>""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
