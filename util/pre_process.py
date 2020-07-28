import re
import numpy as np
import pandas as pd

import spacy
import logging

from nltk.tokenize import WhitespaceTokenizer
from nltk.stem import WordNetLemmatizer

import json

contraction_dict = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", 
                        "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", 
                        "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", 
                        "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  
                        "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", 
                        "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have",
                        "i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", 
                        "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam",
                        "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have",
                        "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not",
                        "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not",
                        "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", 
                        "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", 
                        "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have",
                        "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", 
                        "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", 
                        "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is",
                        "they'd": "they would", "they'd've": "they would have", "they'll": "they will", 
                        "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", 
                        "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", 
                        "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", 
                        "what'll": "what will", "what'll've": "what will have", "what're": "what are",  "what's": "what is",
                        "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did",
                        "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", 
                        "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", 
                        "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not",
                        "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would",
                        "y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have",
                        "you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", 
                        "you're": "you are", "you've": "you have"}
        
def _get_contractions(contraction_dict):
    contraction_re = re.compile('(%s)' % '|'.join(contraction_dict.keys()))
    return contraction_dict, contraction_re

contractions, contractions_re = _get_contractions(contraction_dict)

def replace_contractions(x):
    def replace(match):
        return contractions[match.group(0)]
    return contractions_re.sub(replace, str(x))

def replace_contractions_series(series):    
    series = series.copy()
    series = series.apply(replace_contractions)
    return series

def clean_numbers_series(series):
    if isinstance(series, pd.Series):
        series = series.copy()
        series = series.str.replace('[0-9]{6,}', '######')
        series = series.str.replace('[0-9]{5,}', '#####')
        series = series.str.replace('[0-9]{4,}', '####')
        series = series.str.replace('[0-9]{3,}', '###')
        series = series.str.replace('[0-9]{2,}', '##')
        series = series.str.replace('[0-9]{1,}', '#')
        return series
    else:
        raise ValueError("Need pandas series as input")

def lemmatize_series(series, lematize=False, spacy=False):
    if isinstance(series, pd.Series):
        series = series.copy()
        w_tokenizer = WhitespaceTokenizer()
        lemmatizer = WordNetLemmatizer()
        tokenize_lematize_word_list = []
        for i in list(series):
            if lematize:
                if spacy==True:
                    doc = nlp(i)
                    tokenize_lematize_word_list.append([token.lemma_ for token in doc])
                else:
                    tokenize_lematize_word_list.append([lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(i)])
            else: 
                tokenize_lematize_word_list.append([w for w in w_tokenizer.tokenize(i)])
        return pd.Series(tokenize_lematize_word_list)
    else:
        raise ValueError("Need pandas series as input")

def stopword_removal_series(series):
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
    nlp.Defaults.stop_words |= {"s"}
    stop_words = [i for i in nlp.Defaults.stop_words]

    if isinstance(series, pd.Series):
        series = series.copy()
        stop_word_processed_list = []
        for i in list(series):
            stop_word_processed_list.append([item for item in i if item not in stop_words])
        return stop_word_processed_list
    else:
        raise ValueError("Need pandas series as input")

def preprocess_text(series, lower=True, remove_digits=False, clean_numbers=True, replace_contractions=True, 
                    remove_nonchars=True, lematize=True, stop_word_removal=True, lema_with_spacy=False):
    
    series = series.copy()
    
    if lower:
        series = series.str.lower()
        
    if remove_digits:
        series = series.str.replace(r"\d", "")
    
    if replace_contractions:
        series = replace_contractions_series(series)
    
    if remove_nonchars:
        series = series.str.replace(r"[^a-z0-9 ]+", " ")
        
    if clean_numbers:
        series = clean_numbers_series(series)    
    
    series = series.str.replace(' +',' ',)
    
    if lematize:
        series = lemmatize_series(series, lematize=lematize, spacy=lema_with_spacy)
        
    if stop_word_removal:
        series = stopword_removal_series(series)
      
        
    return series


def extract_names(x):
    tlist = []
    try:
        t0 = json.loads(x)
        for t1 in t0:
            t2 = t1['name'].lower().replace(" ", "")
            t2 = t2.replace("-", "")
            tlist.append(t2)
            #tlist.append(t1['name'].lower().replace(" ", ""))
            #tlist.append(t1['name'].lower()) 
    except:
         pass
    return ','.join(tlist)

def spacy_meta_preprocess(row, spacy_nlp):
    try:
        doc = spacy_nlp(row['master_meta'])
        tokens = [token.lemma_ for token in doc if not token.is_stop]
        tokens = [token.replace(" ", "") for token in tokens if len(token.replace(" ", "")) > 2]
        tokens = list(set(tokens))
        return tokens
    except:
        pass
    return np.NaN


def preprocess_meta(df, min_nwords=5):
    try:
        logging.info('preprocess_meta starts')
        df['entity_data_asset_meta_language'] = df['entity_data_asset_meta_language'].str.lower()
        df['entity_data_asset_meta_genre_name'] = df['entity_data_asset_meta_genre_name'].str.lower()
        df['entity_data_asset_meta_subgenre_name'] = df['entity_data_asset_meta_subgenre_name'].str.lower()
        df['entity_data_asset_meta_subsubgenre_name'] = df['entity_data_asset_meta_subsubgenre_name'].str.lower()
        df['entity_data_asset_meta_tags'] = df['entity_data_asset_meta_tags'].str.lower()
        
        logging.info('preprocess_meta lower conversion done')

        df['entity_data_asset_meta_directors_name'] = df['entity_data_asset_meta_directors'].apply(extract_names)
        df['entity_data_asset_meta_actors_name'] = df['entity_data_asset_meta_actors'].apply(extract_names)
        df['entity_data_asset_meta_actresses_name'] = df['entity_data_asset_meta_actresses'].apply(extract_names)
        df['entity_data_asset_meta_producers_name'] = df['entity_data_asset_meta_producers'].apply(extract_names)
        logging.info('preprocess_meta extract done')

        df['master_meta'] = df[['entity_data_asset_meta_tags', 
                                'entity_data_asset_meta_language', 
                                'entity_data_asset_meta_genre_name', 
                                'entity_data_asset_meta_subgenre_name',
                                'entity_data_asset_meta_subsubgenre_name',                                
                                'entity_data_asset_meta_actors_name',
                                'entity_data_asset_meta_actresses_name',
                                'entity_data_asset_meta_directors_name',
                                'entity_data_asset_meta_producers_name']].apply(lambda s: s.str.cat(sep=','), axis=1)

        df['master_meta'] = df['master_meta'].str.replace(r"\d", "")
        df['master_meta'] = df['master_meta'].str.replace(r"[^a-z0-9 ]+", " ")

        spacy_nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
        customize_stop_words = ['television', 'drama', 'korean', 'movie', 'hindi', 'episode']
        for w in customize_stop_words:
            spacy_nlp.vocab[w].is_stop = True
        
        logging.info('preprocess_meta spacy preprocess starts')
        df['master_meta_split'] = df.apply(spacy_meta_preprocess, spacy_nlp=spacy_nlp, axis=1)
        logging.info('preprocess_meta spacy preprocess ends')

        df['master_meta_split_len'] = df['master_meta_split'].str.len()
        df = df.loc[df['master_meta_split_len']>=min_nwords]
        df.reset_index(inplace=True)
        logging.info('preprocess_meta master meta done')
    except:
        return False, None

    return True, df