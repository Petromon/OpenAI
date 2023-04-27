import urllib.request as req_pdf
import requests
from bs4 import BeautifulSoup
import re
import numpy as np
import tensorflow_hub as hub
import openai
import gradio as gr
import os
from sklearn.neighbors import NearestNeighbors
import tensorflow as tf
from pypdf import PdfReader

org_url_web = ''
org_url_pdf = ''

def download_pdf(url, output_path):
    req_pdf.urlretrieve(url, output_path)

def preprocess(text):
    text = text.replace('\n', ' ')
    text = re.sub('\s+', ' ', text)
    return text


def pdf_to_text(path, start_page=1, end_page=None):
    
    text_list = []

    reader = PdfReader(path)
    total_pages = len(reader.pages)
    if end_page is None:
        end_page = total_pages

    for i in range(start_page-1, end_page):
        page = reader.pages[i]
        text = preprocess(page.extract_text())
        text_list.append(text)

    os.remove(path)
    
    return text_list


def text_to_chunks(texts, word_length=150, start_page=1):
    text_toks = [t.split(' ') for t in texts]
    chunks = []
    
    for idx, words in enumerate(text_toks):
        for i in range(0, len(words), word_length):
            chunk = words[i:i+word_length]
            if (i+word_length) > len(words) and (len(chunk) < word_length) and (len(text_toks) != (idx+1)):
                text_toks[idx+1] = chunk + text_toks[idx+1]
                continue
            chunk = ' '.join(chunk).strip()
            chunk = f'[{idx+start_page}]' + ' ' + '"' + chunk + '"'
            chunks.append(chunk)
    return chunks


class SemanticSearch:
    
    def __init__(self):
        self.use = hub.load('https://tfhub.dev/google/universal-sentence-encoder/4')
        self.fitted = False
    
    def fit(self, data, batch=1000, n_neighbors=5):
        self.data = data
        self.embeddings = self.get_text_embedding(data, batch=batch)
        n_neighbors = min(n_neighbors, len(self.embeddings))
        self.nn = NearestNeighbors(n_neighbors=n_neighbors)
        self.nn.fit(self.embeddings)
        self.fitted = True
    
    def __call__(self, text, return_data=True):
        inp_emb = self.use([text])
        
        neighbors = self.nn.kneighbors(inp_emb, return_distance=False)[0]
        
        if return_data:
            return [self.data[i] for i in neighbors]
        else:
            return neighbors
    
    def get_text_embedding(self, texts, batch=1000):
        embeddings = []
        for i in range(0, len(texts), batch):
            text_batch = texts[i:(i+batch)]
            emb_batch = self.use(text_batch)
            embeddings.append(emb_batch)
        embeddings = np.vstack(embeddings)
        return embeddings

def scrap_website(url):
    # Send GET request to webpage
    response = requests.get(url)

    # Parse HTML content with BeautifulSoup
    soup = BeautifulSoup(response.content, 'html.parser')

    # Extract webpage text using BeautifulSoup
    page_text = preprocess(soup.get_text())
    
    # Split text into chunks of 4000 tokens or less
    # chunks = [page_text[i:i+4000] for i in range(0, len(page_text), 4000)]
    return page_text

# def load_recommender(path, start_page=1):
#     global recommender
#     texts = ''
#     texts = pdf_to_text(path, start_page=start_page)
#     return texts


def generate_text(openAI_key,prompt, engine="text-davinci-003"):
    openai.api_key = openAI_key

    # Split text into chunks of 4000 tokens or less
    chunks = [prompt[i:i+4000] for i in range(0, len(prompt), 4000)]

    # Use OpenAI GPT-3 to analyze text chunks
    for chunk in chunks:
        completions = openai.Completion.create(
            engine=engine,
            prompt=chunk,
            max_tokens=512,
            n=1,
            stop=None,
            temperature=0.7,
        )
    message = completions.choices[0].text
    return message


def generate_answer(recommender, question, openAI_key):
    topn_chunks = []
    if hasattr(recommender, 'data'):
        topn_chunks = recommender(question)
    else:
        topn_chunks.append(question)
    prompt = ""
    # prompt += 'search results:\n\n'
    prompt += 'Hakutulokset:\n\n'
    for c in topn_chunks:
        prompt += c + '\n\n'
        
    # prompt += "Instructions: Compose a comprehensive reply to the query using the search results given. "\
    #           "Cite each reference using [number] notation (every result has this number at the beginning). "\
    #           "Citation should be done at the end of each sentence. If the search results mention multiple subjects "\
    #           "with the same name, create separate answers for each. Only include information found in the results and "\
    #           "don't add any additional information. Make sure the answer is correct and don't output false content. "\
    #           "If the text does not relate to the query, simply state 'Found Nothing'. Ignore outlier "\
    #           "search results which has nothing to do with the question. Only answer what is asked. The "\
    #           "answer should be short and concise.\n\nQuery: {question}\nAnswer: "
    
    prompt += "Ohjeet: Kirjoita kyselyyn kattava vastaus annettujen hakutulosten avulla."\
              "Lainaa jokainen viite käyttämällä [numero]-merkintää (jokaisen tuloksen alussa on tämä numero). "\
              "Lainaus tulee tehdä jokaisen lauseen loppuun. Jos hakutuloksissa mainitaan useita samannimiä aiheita, luo jokaiselle oma vastaus."\
              "Sisällytä vain tuloksista löydetyt tiedot, äläkä lisää muita tietoja."\
              "Varmista, että vastaus on oikea, äläkä anna väärää sisältöä."\
              "Jos teksti ei liity kyselyyn, ilmoita vain 'Ei löytynyt mitään'."\
              "Ohita poikkeavia hakutuloksia, joilla ei ole mitään tekemistä kysymyksen kanssa."\
              "Vastaa vain siihen, mitä kysytään."\
              "Vastauksen tulee olla lyhyt ja ytimekäs.\n\nQuery: {question}\nAnswer: "\
              
              
    prompt += f"Query: {question}\nAnswer:"
    answer = generate_text(openAI_key, prompt,"text-davinci-003")
    return answer


def question_answer(url_web, url_pdf, file, question,openAI_key):
    # global recommender
    global org_url_pdf
    global org_url_web
    recommender = SemanticSearch()

    if openAI_key.strip()=='':
        return '[ERROR]: Please enter you Open AI Key. Get your key here : https://platform.openai.com/account/api-keys'
    if url_pdf.strip() == '' and file == None and url_web.strip() == '':
        return '[ERROR]: Both URL and PDF is empty. Provide atleast one.'
    
    if url_pdf.strip() != '' and file != None:
        return '[ERROR]: Both URL and PDF is provided. Please provide only one (eiter URL or PDF).'
    texts = []
    #download and save 'corpus.pdf'
    if url_pdf.strip() != '':
        if org_url_pdf != url_pdf:
            download_pdf(url_pdf, 'corpus.pdf')
            org_url_pdf = url_pdf
            # load_recommender('corpus.pdf', url_web)
            texts = pdf_to_text('corpus.pdf', start_page=1)
    else:
        if org_url_pdf != url_pdf:
            org_url_pdf = url_pdf
            old_file_name = file.name
            file_name = file.name
            file_name = file_name[:-12] + file_name[-4:]
            os.rename(old_file_name, file_name)
            # load_recommender(file_name, url_web)
            texts = pdf_to_text(file_name, start_page=1)
    if org_url_web != url_web:
        texts.append(scrap_website(url_web))
        org_url_web = url_web
    chunks = text_to_chunks(texts, start_page=1)
    if len(chunks) > 0:
        recommender.fit(chunks)
    
    if question.strip() == '':
        return '[ERROR]: Question field is empty'

    return generate_answer(recommender, question, openAI_key)

title = 'Scraping, Fine-tuning web site and pdf documents'
description = """<p style="text-align:center">GPT-4 Chatbot integration for customer support</p>"""

with gr.Blocks() as demo:

    gr.Markdown(f'<center><h1>{title}</h1></center>')
    gr.Markdown(description)

    with gr.Row():
        
        with gr.Group():
            gr.Markdown(f'<p style="text-align:center">Get your Open AI API key <a href="https://platform.openai.com/account/api-keys">here</a></p>')
            openAI_key=gr.Textbox(label='Enter your OpenAI API key here')
            url_web = gr.Textbox(label='Enter WEB SITE URL here')
            url_pdf = gr.Textbox(label='Enter PDF URL here')
            gr.Markdown("<center><h4>OR<h4></center>")
            file = gr.File(label='Upload your PDF/ Research Paper / Book here', file_types=['.pdf'])
            question = gr.Textbox(label='Enter your question here')
            btn = gr.Button(value='Submit')
            btn.style(full_width=True)

        with gr.Group():
            answer = gr.Textbox(label='The answer to your question is :')

        btn.click(question_answer, inputs=[url_web, url_pdf, file, question,openAI_key], outputs=[answer])
#openai.api_key = os.getenv('Your_Key_Here') 
demo.launch(share=True)


# import streamlit as st

# #Define the app layout
# st.markdown(f'<center><h1>{title}</h1></center>', unsafe_allow_html=True)
# st.markdown(description)

# col1, col2 = st.columns(2)

# # Define the inputs in the first column
# with col1:
#     url = st.text_input('URL')
#     st.markdown("<center><h6>or<h6></center>", unsafe_allow_html=True)
#     file = st.file_uploader('PDF', type='pdf')
#     question = st.text_input('question')
#     btn = st.button('Submit')

# # Define the output in the second column
# with col2:
#     answer = st.text_input('answer')

# # Define the button action
# if btn:
#     answer_value = question_answer(url, file, question)
#     answer.value = answer_value