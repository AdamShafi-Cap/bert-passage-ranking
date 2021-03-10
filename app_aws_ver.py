# Imports
import pandas as pd
import numpy as np
import pickle
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
from summarizer import Summarizer, sentence_handler
#import tensorflow_hub as hub

# BERT based models for document search
from sentence_transformers import SentenceTransformer


### App Set Up ###

stqdm.pandas()
st.set_page_config(layout="wide")
file, text, q, embeddings_option, num_pages = None, None, None, None, None

st.title('Project Pico: Natural Language Processing (NLP) Demo')
st.header('Applications of BERT Models')
st.write('''
BERT is an open-source model for processing natural language, developed by Google. It was designed to help computers understand the meaning of ambiguous language in text by using surrounding text to establish context.''')
st.write('''
This app uses BERT models trained to specific tasks to demonstrate the model's ability to a) find relevant passages in a document, given a query and b) find representative sentences within the passage to create a summary. To use it, either upload a document or use the example Brexit document. You must then select a model from the list of BERT variants and enter a query. The app will use the model you select to search the query against the document. It then uses a separate BERT model to write a summary using sentences from the paragraph.''' )
st.write('-'*50)

model_desc = {
            '':'''Once you make a selection, the model will process your document. This can take some time if your document is large. For fastest processing, use the Brexit document and one of the DistillBERT models. All models in this list were fine tuned by researchers at the 
Technical University of Darmstadt.''',
           'DistillBERT':'''
Developed by Huggingface as a faster, more efficient version of BERT using a compression technique called distillation. 
In this app, this model is fine tuned on datasets aimed at training models to find similar text.
           ''', 
           'RoBERTa Large':'''
Developed by Facebook as a larger, more optimized version of BERT. When released, this model improved BERT’s scores on a variety of benchmarks. 
In this app, this model is fine tuned on datasets aimed at training models to find similar text.
           ''',
           'DistillBERT Q&A':'''
Developed by Huggingface as a faster, more efficient version of BERT using a compression technique called distillation.
This version is fine tuned on Microsoft’s Q&A dataset  which includes millions of question and answer pairs from web search.
           '''
            }

### Define Functions ###

@st.cache(allow_output_mutation=True)
def load_models():   
    try:
        a = pickle.load(open('./models/dbert.pkl', 'rb'))
    except:
        a = SentenceTransformer('stsb-distilbert-base')
        pickle.dump(a, open('./models/dbert.pkl', 'wb'))
    try:
        b = pickle.load(open('./models/rbert.pkl', 'rb'))
    except:
        b = SentenceTransformer('stsb-roberta-large')
        pickle.dump(b, open('./models/rbert.pkl', 'wb'))   
    try:
        c = pickle.load(open('./models/qbert.pkl', 'rb'))
    except:
        c = SentenceTransformer('msmarco-distilbert-base-v2')
        pickle.dump(c, open('./models/qbert.pkl', 'wb'))       
    return a,b,c   


@st.cache(allow_output_mutation=True)#hash_funcs={preshed.maps.PreshMap:id, cymem.cymem.Pool:id}, 
def load_summarizer():
    try:
        summ = pickle.load(open('./models/summarizer.pkl', 'rb'))
    except:
        summ = Summarizer()
        pickle.dump(summ, open('./models/summarizer.pkl', 'wb'))
    return summ

@st.cache()
def load_pdf(file,n=0)->str:
    
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
    for i, page in enumerate(PDFPage.get_pages(fp)):
        if i+1 > n:
            interpreter.process_page(page)
    text = retstr.getvalue()
    return text

@st.cache()
def get_articles_brexit(text:str)->pd.Series:
    
    data = pd.Series(text.split('\n\nARTICLE'))

    s = (data
         .str.strip()
         .loc[46:]
         .loc[lambda x: x.astype(bool)]
         .loc[lambda x: x.apply(len)>10]
         .str.replace('\s+',' ')
         #.str.replace('\n\n','\n')
         #.str.replace(r'\.\s*$', '. <br />')
         .drop_duplicates()
        )
    
    return s

@st.cache()
def get_articles(text:str)->pd.DataFrame:
    
    data = text.split('\x0c')
    
    df = pd.DataFrame(enumerate(data,1),columns=['page','text'])
    
    df = df.assign(text=df['text'].str.split('\n\n')).explode('text').reset_index(drop=True)
    df['text'] = df['text'].str.strip().str.replace('\s+',' ')
    df['words'] = df['text'].apply(lambda x: len(x.split(' ')))
    df = df.loc[lambda x: x.text.astype(bool)].loc[lambda x: x.words>20].drop_duplicates(subset='text')
    df.drop('words',axis=1,inplace=True)
    
    return df    
    
@st.cache()
def ask(q:str, X:pd.DataFrame, s:pd.DataFrame, n: int, model)->pd.Series:
    
    embedding = np.array(model.encode([q])[0])
        
    sorted_index = (X
                    .apply(lambda row: np.dot(row, embedding), axis=1)
                    .abs()
                    .sort_values(ascending=False)
                   )
    
    return s.loc[sorted_index.index].head(n)

@st.cache()
def summarize(text, n=1):
    result = summarizer_model(text, num_sentences=n)
    return result

@st.cache(suppress_st_warning=True)
def get_embeddings(s:pd.Series, embeddings_option)->pd.DataFrame:
    model=options[embeddings_option][1]
    return s.progress_apply(model.encode).apply(pd.Series)


def get_embeddings_brexit(embeddings_option):
    with open(options[embeddings_option][0], 'rb') as file:
        ans = pickle.load(file)
    return ans

def bold_sentences(text,summary):
    handler =  sentence_handler.SentenceHandler()
    bold = " ".join([f"__**{sentence}**__" 
                    if summary.find(sentence) != -1
                    else sentence 
                    for sentence in handler.process(text,min_length = 0)])
    return bold


### App ###

### Load Models ###
dbert,rbert,qbert = load_models()
summarizer_model = load_summarizer()

options = {
           '':['',''],
           'DistillBERT':['./embeddings/distilbert.pkl',dbert], 
           'RoBERTa Large':['./embeddings/robert.pkl',rbert],
           'DistillBERT Q&A':['./embeddings/distilbertqa.pkl',qbert]
          }

selectbox_list = list(options.keys())


### Choose Method ###
col1, col2 = st.beta_columns(2)
with col1:
    method = st.radio('Choose one...',['Upload a PDF','Upload a csv','Brexit Trade Agreement'])

### PDF Document ###

if method == 'Upload a PDF':
    file=None
    with col1:
        st.write('Upload a PDF document which will the app will attempt to split into paragraphs. You can remove the first few pages, using the button to the right, to prevent titles and contents pages from being included. A good document size is 40-60 pages and larger documents will take longer to process. ')
    with col2:
        num_pages=0
        if st.checkbox('Remove pages from start of document?'):
            num_pages = st.number_input('Enter number of pages to remove',value=0,step=1)
        file = st.file_uploader('Upload your document.')

    if file:
        try:
            text = load_pdf(file,num_pages)
            s = get_articles(text)
        except: 
            col2.warning('Invalid PDF type, or too many pages removed. Try a different document or remove fewer pages')
            file=None
            st.stop()

    
        st.write('-'*50)
        col3, col4 = st.beta_columns(2)

        with col3:
            embeddings_option = st.selectbox('Which model?', selectbox_list, index=0)

        with col4:
            st.write(f"About {embeddings_option}:")
            st.write(model_desc[embeddings_option])
        
            
        if embeddings_option!=selectbox_list[0]:
            X = get_embeddings(s.iloc[:,1],embeddings_option=embeddings_option)


            q = st.text_input('What is your query?')

        if q:
            ans = ask(q, X=X, s=s, n=3, model=options[embeddings_option][1])
            for i,t in ans.values:
                with st.beta_expander(f'PAGE {i}'):
                    if len(t)>40:
                        summary = summarize(t, 1)
                        st.success(summary)
                        st.write(bold_sentences(t,summary))
                    else:
                        st.write(t)
    
### CSV Document ###

elif method == 'Upload a csv':
    file=None
    with col1:
        st.write('Upload a csv file with 1 column and no column names, each row should contain a separate paragraph. The model will match your query to each row and return the best matches. A good size is 200-400 rows.')
    with col2:
        file = st.file_uploader('Upload your document.')
    if file:
        try:
            s = pd.read_csv(file,header=None,keep_default_na=False)
            s.index=s.index+1
            s=s.iloc[:, 0].str.strip()
        except: 
            col2.warning('Invalid csv, try a different document')
            file=None
            st.stop()
            
        st.write('-'*50)
        
        col3, col4 = st.beta_columns(2)

        with col3:
            embeddings_option = st.selectbox('Which model?', selectbox_list)

        with col4:
            st.write(f"About {embeddings_option}:")
            st.write(model_desc[embeddings_option])

        if embeddings_option!=selectbox_list[0]:
            X = get_embeddings(s,embeddings_option=embeddings_option)               

            q = st.text_input('What is your query?')


        if q:
            ans = ask(q, X=X, s=s, n=3, model=options[embeddings_option][1])
            for i, t in zip(ans.index,ans):
                with st.beta_expander(f'ROW NUMBER: {i}'):
                    if len(t)>50:
                        summary = summarize(t, 1)
                        st.success(summary)
                        st.write(bold_sentences(t,summary))
                    else:
                        st.write(t)
                           



### Brexit Document ###
    
elif method == 'Brexit Trade Agreement':
    file=None
    file = 'DRAFT_UK-EU_Comprehensive_Free_Trade_Agreement.pdf'
    st.write('This will use a draft version of the Brexit trade agreement. This has been selected because it is long and uses domain-specific language. Although this document is over 200 pages, it has been processed in advance to ensure good performance in this app. You can view the original document here: https://assets.publishing.service.gov.uk/government/uploads/system/uploads/attachment_data/file/886010/DRAFT_UK-EU_Comprehensive_Free_Trade_Agreement.pdf')
    text = load_pdf(file)
    s = get_articles_brexit(text)

    col3, col4 = st.beta_columns(2)
    with col3:
        embeddings_option = st.selectbox('Which model?', selectbox_list[1:], index=0)
    with col4:
        st.write(f"About {embeddings_option}:")
        st.write(model_desc[embeddings_option])

    q = st.text_input('What is your query?')


    if q:
        X = get_embeddings_brexit(embeddings_option)
        ans = ask(q, X=X, s=s, n=3, model=options[embeddings_option][1])
        for i, t in enumerate(ans):
            with st.beta_expander(f'ARTICLE {t.split()[0]}'):
                if len(t)>40:
                    summary = summarize(t, 1)
                    st.success(summary)
                    t = bold_sentences(t,summary)  
                    st.write(f"ARTICLE {t}")
                else:
                    st.write(f"ARTICLE {t}")
                
                
                
else:
    st.write('ERROR')