# Imports
import pandas as pd
import numpy as np
import tensorflow_hub as hub
from stqdm import stqdm

# Streamlit
import streamlit as st
import preshed
import cymem

# PDF
import sys
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from pdfminer.converter import XMLConverter, HTMLConverter, TextConverter
from pdfminer.layout import LAParams
import io

# Summarization using extractive bert
from summarizer import Summarizer

# BERT based models for document search
from sentence_transformers import SentenceTransformer
import pickle

# GPT-2
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer
import tensorflow as tf

st.set_page_config(layout="wide")
file, text, q = None, None, None
stqdm.pandas()


@st.cache(allow_output_mutation=True)
def load_models():

    
    #a = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    a = None

    try:
        b = pickle.load(open('./models/dbert.pkl', 'rb'))
    except:
        b = SentenceTransformer('stsb-distilbert-base')
        pickle.dump(b, open('./models/dbert.pkl', 'wb'))

    try:
        c = pickle.load(open('./models/rbert.pkl', 'rb'))
    except:
        c = SentenceTransformer('stsb-roberta-large')
        pickle.dump(c, open('./models/rbert.pkl', 'wb'))   

    try:
        d = pickle.load(open('./models/qbert.pkl', 'rb'))
    except:
        d = SentenceTransformer('msmarco-distilbert-base-v2')
        pickle.dump(d, open('./models/qbert.pkl', 'wb'))       
    return a,b,c,d   


@st.cache(hash_funcs={preshed.maps.PreshMap:id, cymem.cymem.Pool:id}, allow_output_mutation=True)#hash_funcs={preshed.maps.PreshMap: lambda x: 1, cymem.cymem.Pool:      lambda x: 1})
def load_summarizer():
    try:
        model = pickle.load(open('./models/summarizer.pkl', 'rb'))
    except:
        model = Summarizer()
        pickle.dump(model, open('./models/summarizer.pkl', 'wb'))
    return model

@st.cache()
def load_pdf(file)->str:
    
    if isinstance(file, str):
        fp = open(file, 'rb')
    else: 
        fp = file
        
    rsrcmgr = PDFResourceManager()
    retstr = io.StringIO()
    laparams = LAParams()
    device = TextConverter(rsrcmgr, retstr, laparams=laparams)
    
    # Create a PDF interpreter object.
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    
    # Process each page contained in the document.
    pages = []
    for i, page in enumerate(PDFPage.get_pages(fp)):
        interpreter.process_page(page)
        text = retstr.getvalue()
        pages.append(text)
        
    full_text = pages[-1]
    return full_text

@st.cache()
def get_articles(text:str)->pd.Series:
    
    data = pd.Series(text.split('\n\nARTICLE'))

    s = (data
         .str.strip()
         .loc[46:]
         .loc[lambda x: x.astype(bool)]
         .loc[lambda x: x.apply(len)>10]
         .str.replace('\s+',' ')
         .drop_duplicates()
        )
    
    return s

    
    
@st.cache()
def ask(q:str, X:pd.DataFrame, s:pd.Series, n: int, model, embeddings_option)->pd.Series:
    
    if embeddings_option == selectbox_list[0]:
        embedding = np.array(model([q])[0])
    else:
        embedding = np.array(model.encode([q])[0])
        
    sorted_index = (X
                    .apply(lambda row: np.dot(row, embedding), axis=1)
                    .abs()
                    .sort_values(ascending=False)
                   )
    
    return s.loc[sorted_index.index].head(n)

#@st.cache()
def summarize(text, model, n=1):
    result = model(text, num_sentences=n)
    return result

def get_embeddings(embeddings_option):
    

    return pd.read_pickle(options[embeddings_option][0])


model_gpt = TFGPT2LMHeadModel.from_pretrained('distilgpt2')
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

def ab_sum(q,t):
    in_text = t+' '+q
    input_ids = tokenizer.encode(in_text, return_tensors="tf")
    max_len = input_ids.shape[1]
    start = len(in_text)
    generated_text_samples = model_gpt.generate(
                                            input_ids, 
                                            max_length=max_len+20,  
                                            num_return_sequences=1,
                                            no_repeat_ngram_size=2,
                                            repetition_penalty=1.5,
                                            top_p=0.92,
                                            temperature=.85,
                                            do_sample=True,
                                            top_k=125,
                                            early_stopping=True)
    return tokenizer.decode(generated_text_samples[0], skip_special_tokens=True)[start:]

### APP
st.title('BERT Passage Scoring')

# ALWAYS
use,dbert,rbert,qbert = load_models()


options = {'Universal Sentence Encoder': ['./embeddings/use.pkl',use],
           'DistillBERT':['./embeddings/distilbert.pkl',dbert], 
           'RoBERTa Large':['./embeddings/robert.pkl',rbert],
           'DistillBERT Q&A':['./embeddings/distilbertqa.pkl',qbert],
           'Select a model...':['','']}

selectbox_list = list(options.keys())

summarizer_model = load_summarizer()



    
file = 'DRAFT_UK-EU_Comprehensive_Free_Trade_Agreement.pdf'

text = load_pdf(file)
s = get_articles(text)
embeddings_option = st.selectbox('Which model?', selectbox_list, index=4)
q = st.text_input('What is your query?')


if q:
    X = get_embeddings(embeddings_option)
    ans = ask(q, X=X, s=s, n=3, model=options[embeddings_option][1],embeddings_option=embeddings_option)
    for i, t in enumerate(ans):
        with st.beta_expander(f'ARTICLE {t.split()[0]}'):
            if len(t.split('. '))>3:
                summary = summarize(t, summarizer_model, 1)
                ab_summary = ab_sum(q,t)
                st.success(summary)
               #t = ". ".join([f"**{sent}**" if sent in summary.split(". ") else sent for sent in t.split(". ")])   
            else: 
                ab_summary=None
            st.write(t)
            if ab_summary:
                st.warning(ab_summary)

