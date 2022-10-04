# ---------------------------------- import package --------------------------------- #

import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt
import io
from wordcloud import WordCloud
import string
import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
import warnings
import collections
import altair as alt

warnings.filterwarnings("ignore")
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('omw-1.4')

# cd C:\Users\424714\Desktop\WordCloud
# py -m streamlit run app.py

# ---------------------------------- basic parameters --------------------------------- #

st.set_page_config(
    page_title = "Word Cloud Generator",
    # page_icon = r"https://upload.cc/i1/2021/12/31/RtxJjG.png",
    layout = "wide",
    initial_sidebar_state="expanded",
    menu_items = {
        'About': "# Created by Jack Wang.\n Linkedin: https://www.linkedin.com/in/weichieh1998/"
     }
)

hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)

# ---------------------------------- Sidebar --------------------------------- #

st.sidebar.title("Word Cloud Generator")
st.sidebar.markdown("""- Created by
    <a href="https://www.linkedin.com/in/weichieh1998/">Jack Wang</a>"""
    ,
    unsafe_allow_html=True,
    )

# ---------------------------------- Step 1: Upload Files --------------------------------- #

st.header('Step 1: Upload File')

uploaded_file = st.file_uploader(
    label = "Upload excel or csv file (The file must contain only one sheet.)",
    type = ["xlsx","csv"],
    accept_multiple_files = False,
    help = '''File must be csv or excel format.''')

if uploaded_file is not None:
    if uploaded_file.name.endswith('csv'):
        df = pd.read_csv(uploaded_file).astype('str')
    elif uploaded_file.name.endswith('xlsx'):
        df = pd.read_excel(uploaded_file).astype('str')

    st.dataframe(df)
    st.caption(f'Total: {len(df)}')

# ---------------------------------- Step 2: Add filter --------------------------------- #

st.header('Step 2: Add Filter')
ft_check = st.checkbox(label='Add Filter', value=True)

try:
    if ft_check:
        options = df.columns
        filter_col = st.selectbox(
            label="Select one column you want to filter",
            options=options)

        options = df[filter_col].unique()
        filter = st.multiselect(
            label="Select values you want to filter",
            options=options)

        df = df[df[filter_col].isin(filter)]
        st.caption(f'Total: {len(df)}')
except:
    pass

# ---------------------------------- Step 3: Select Columns --------------------------------- #

st.header('Step 3: Select Column')

try:
    options = df.columns
    column = st.selectbox(
        label = "Select one column you want to analyze",
        options = options
    )

    # Preprocessing
    df = df[column].dropna().to_frame().reset_index(drop=True).astype('str')
    text = [text.strip() for text in df[column]]
    text = ' '.join(text)
    no_punc_text = text.translate(str.maketrans('','',string.punctuation))
    text_tokens = word_tokenize(no_punc_text)

except:
    pass

# ---------------------------------- Step 4: Advanced options --------------------------------- #

st.header('Step 4: Advanced Options')

# Predifined stopwords
myfile = requests.get('https://github.com/wcwang1998/word_cloud_generator/blob/main/StopWords.xlsx?raw=true')
my_stop_words = pd.read_excel(myfile.content)['StopWord'].tolist()
sw_list = ['I','The','It','A','”',"’",'“','wasn','nan']
my_stop_words.extend(sw_list)

stopwords = st.checkbox(label='Remove stop words', value=True)

# More stopwords
try:
    more_sw = st.multiselect('Enter other keywords you want to remove',
                             options=list(set([x.lower() for x in text_tokens])))
    my_stop_words.extend(more_sw)
except:
    pass

# ---------------------------------- Button: Generate Word Cloud --------------------------------- #

columns = st.columns((2, 1, 2))
start = columns[1].button('Generate Word Cloud')
st.markdown("----", unsafe_allow_html=True)

try:
    lower_words = [comment.lower() for comment in text_tokens]
except:
    pass

def wordcloud(no_stop_tokens):
    wl = WordNetLemmatizer()

    # This is a helper function to map NTLK position tags
    def get_wordnet_pos(tag):
        if tag.startswith('J'):
            return wordnet.ADJ
        elif tag.startswith('V'):
            return wordnet.VERB
        elif tag.startswith('N'):
            return wordnet.NOUN
        elif tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN

    # Tokenize the sentence
    def lemmatizer(string):
        word_pos_tags = nltk.pos_tag(word_tokenize(string))  # Get position tags
        a = [wl.lemmatize(tag[0], get_wordnet_pos(tag[1])) for idx, tag in
             enumerate(word_pos_tags)]  # Map the position tag and lemmatize the word/token
        return " ".join(a)

    def finalpreprocess(string):
        return lemmatizer(string)

    # Correct spelling
    text_correction = [finalpreprocess(string) for string in no_stop_tokens]

    wordcloud = WordCloud(width=2000, height=1500, max_words=100, colormap='Dark2', background_color="white",
                          stopwords=my_stop_words).generate(" ".join(text_correction))
    fig = plt.figure(figsize=(20, 15))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")

    return fig, text_correction

with st.spinner('Generating...'):
    if stopwords and start:
        no_stop_tokens = [word for word in lower_words if not word in my_stop_words]
        fig, text_correction = wordcloud(no_stop_tokens)
    elif start:
        no_stop_tokens = lower_words
        fig, text_correction = wordcloud(no_stop_tokens)

    try:
        plot = st.pyplot(fig)
    except:
        pass

# ---------------------------------- Button: Download --------------------------------- #

try:
    if plot is not None:
        img = io.BytesIO()
        plt.savefig(img, format='png')
        columns = st.columns((2, 1, 2))
        button_pressed = columns[1].download_button(label="Download Word Cloud",
                                                    data=img,
                                                    file_name='Word Cloud.png',
                                                    mime="image/png")
        st.markdown("----", unsafe_allow_html=True)
except:
    pass

# ---------------------------------- Histogram --------------------------------- #

try:
    frequency = dict(collections.Counter(text_correction))
    frequency = pd.DataFrame(frequency, index=['Frequency']).transpose()
    frequency = frequency[~frequency.index.isin(more_sw)].nlargest(n=30, columns=['Frequency']).reset_index()
    frequency.columns = ['Word','Frequency']
    hist = alt.Chart(frequency.sort_values('Frequency', ascending=False)).mark_bar().encode(x='Frequency',
                    y=alt.Y('Word', sort='-x'), tooltip=['Word', 'Frequency']
                    ).properties(height=600, width=800, title="Top 30 Words By Frequency").interactive()

    st.altair_chart(hist, use_container_width=True)
except:
    pass

