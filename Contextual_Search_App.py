#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


@author: danielsmetana7@ MABA CLASS
"""

import streamlit as st
import pandas as pd
import numpy as np

import plotly.express as px

import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from collections import Counter
from heapq import nlargest
import os
import spacy
nlp = spacy.load("en_core_web_sm")
from spacy import displacy
from string import punctuation
from sentence_transformers import SentenceTransformer
import scipy.spatial
import pickle as pkl

st.title("Daniel Smetana Chicago Contextual Search Engine")

################################################################################

embedder = SentenceTransformer('all-MiniLM-L6-v2')

df = pd.read_csv("hotelReviewsInChicago.csv")

df['hotelName'].drop_duplicates()

df_combined = df.sort_values(['hotelName']).groupby('hotelName', sort=False).review.apply(''.join).reset_index(name='all_review')

df_combined





import re

df_combined['all_review'] = df_combined['all_review'].apply(lambda x: re.sub('[^a-zA-z0-9\s]','',x))

def lower_case(input_str):
    input_str = input_str.lower()
    return input_str

df_combined['all_review']= df_combined['all_review'].apply(lambda x: lower_case(x))

df = df_combined

df_sentences = df_combined.set_index("all_review")

#df_sentences.head()

df_sentences = df_sentences["hotelName"].to_dict()
df_sentences_list = list(df_sentences.keys())

#list(df_sentences.keys())[:5]

import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util

df_sentences_list = [str(d) for d in tqdm(df_sentences_list)]


search_criteria = st.text_input("Please enter search criteria: ")

np.array(search_criteria).reshape(1,-1)


# Corpus with example sentences
corpus = df_sentences_list
corpus_embeddings = embedder.encode(corpus,show_progress_bar=True)

model = SentenceTransformer('all-MiniLM-L6-v2')
queries = search_criteria

paraphrases = util.paraphrase_mining(model, corpus)
query_embeddings_p =  util.paraphrase_mining(model, search_criteria, show_progress_bar=True)


query_embeddings = embedder.encode(search_criteria,show_progress_bar=True)


# Find the closest 5 sentences of the corpus for each query sentence based on cosine similarity
closest_n = 5
st.text("Top 5 most similar hotels:")
for query, query_embedding in zip(search_criteria, query_embeddings):
    distances = scipy.spatial.distance.cdist([query_embedding], corpus_embeddings, "cosine")[0]

    results = zip(range(len(distances)), distances)
    results = sorted(results, key=lambda x: x[1])

    st.text("\n\n=========================================================")
    st.text("==========================Query==============================")
    st.text("===",query,"=====")
    st.text("=========================================================")


    for idx, distance in results[0:closest_n]:
        st.text("Score:   ", "(Score: %.4f)" % (1-distance) , "\n" )
        st.text("Paragraph:   ", corpus[idx].strip(), "\n" )
        row_dict = df.loc[df['all_review']== corpus[idx]]
        st.text("paper_id:  " , row_dict['hotelName'] , "\n")
        # print("Title:  " , row_dict["title"][corpus[idx]] , "\n")
        # print("Abstract:  " , row_dict["abstract"][corpus[idx]] , "\n")
        # print("Abstract_Summary:  " , row_dict["abstract_summary"][corpus[idx]] , "\n")
        st.text("-------------------------------------------")
