from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace  # type: ignore
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter  # type: ignore
from langchain_community.vectorstores import DocArrayInMemorySearch  # type: ignore
from langchain.chains import RetrievalQA, ConversationalRetrievalChain  # type: ignore
from langchain.memory import ConversationBufferMemory  # type: ignore
from langchain_huggingface import HuggingFaceEmbeddings  # type: ignore
from langchain_community.document_loaders import PyPDFLoader  # type: ignore
from langchain_community.document_loaders import TextLoader  # type: ignore
from langchain.prompts import PromptTemplate   # type: ignore
import getpass
import os
import panel as pn  # type: ignore
import param  # type: ignore

import os

import streamlit as st
st.title("My RAG App")
st.write("Running HuggingFace model…")
# Assumes you set this in the Render environment
hf_token = os.environ.get("HUGGINGFACEHUB_API_TOKEN")

if not hf_token:
    raise ValueError("HUGGINGFACEHUB_API_TOKEN environment variable not set.")


def load_db(file, chain_type, k):
    # load documents
    loader = PyPDFLoader(file)
    documents = loader.load()
    # split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.split_documents(documents)
    # define embedding
    model_name = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}

    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    # create vector database from data
    db = DocArrayInMemorySearch.from_documents(docs, embeddings)
    # define retriever
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": k})
    # define the model name
    llm_name = HuggingFaceEndpoint(
        repo_id="microsoft/Phi-3-mini-4k-instruct",
        task="text-generation",
        max_new_tokens=512,
        do_sample=False,
        repetition_penalty=1.03,
    )
    # create a chatbot chain. Memory is managed externally.
    qa = ConversationalRetrievalChain.from_llm(
        llm=ChatHuggingFace(llm=llm_name, verbose=True),
        chain_type=chain_type,
        retriever=retriever,
        return_source_documents=True,
        return_generated_question=True,
    )
    return qa


class cbfs(param.Parameterized):
    chat_history = param.List([])
    answer = param.String("")
    db_query = param.String("")
    db_response = param.List([])

    def __init__(self, **params):
        super(cbfs, self).__init__(**params)
        self.panels = []
        self.loaded_file = None
        self.qa = None
        self.prompt_template = PromptTemplate(
            input_variables=["context"],
            template="""
                You are an intelligent assistant designed to retrieve relevant information from a database and answer user queries accurately.
                Your tasks include:
                1. Retrieving the most relevant data from the given context.
                2. Providing a concise, accurate, and well-structured response to the user's query.
                3. Handling general conversations naturally while staying relevant.

                Context:
                {context}
            
                Based on the context, provide a precise and relevant response. If the context lacks sufficient information, state that explicitly and avoid making assumptions.
                """
        )

    def call_load_db(self, count):
        if count == 0 or file_input.value is None:  # init or no file specified :
            return pn.pane.Markdown(f"Loaded File: {self.loaded_file}")
        else:
            file_input.save("temp.pdf")  # local copy
            self.loaded_file = file_input.filename
            button_load.button_style = "outline"
            self.qa = load_db("temp.pdf", "stuff", 4)
            button_load.button_style = "solid"
        self.clr_history()
        return pn.pane.Markdown(f"Loaded File: {self.loaded_file}")

    def generate_prompt(self, context):
        return self.prompt_template.format(context=context)

    def convchain(self, query):
        if not query:
            return pn.WidgetBox(pn.Row('User:', pn.pane.Markdown("", width=600)), scroll=True)

        if self.qa is None:
            return pn.WidgetBox(
                pn.Row('ChatBot:', pn.pane.Markdown("❗ No document loaded yet. Please upload a PDF first.", width=600)),
                scroll=True
            )

        # Check if the query is already in the chat history
        if self.chat_history and self.chat_history[-1][0] == query:
            return pn.WidgetBox(*self.panels, scroll=True)

        prompt = self.generate_prompt(query)
        result = self.qa.invoke({"question": query, "chat_history": self.chat_history})
        self.chat_history.extend([(query, result["answer"])])
        self.db_query = result["generated_question"]
        self.db_response = result["source_documents"]
        self.answer = result['answer']
        self.panels.extend([
            pn.Row('User:', pn.pane.Markdown(query, width=600)),
            pn.Row('ChatBot:', pn.pane.Markdown(self.answer, width=600))
        ])
        inp.value = ''  # clears loading indicator when cleared
        return pn.WidgetBox(*self.panels, scroll=True)


    @param.depends('db_query', )
    def get_lquest(self):
        if not self.db_query:
            return pn.Column(
                pn.Row(pn.pane.Markdown(f"Last question to DB:", styles={'background-color': '#F6F6F6'})),
                pn.Row(pn.pane.Str("no DB accesses so far"))
            )
        return pn.Column(
            pn.Row(pn.pane.Markdown(f"DB query:", styles={'background-color': '#F6F6F6'})),
            pn.pane.Str(self.db_query)
        )

    @param.depends('db_response', )
    def get_sources(self):
        if not self.db_response:
            return
        rlist = [pn.Row(pn.pane.Markdown(f"Result of DB lookup:", styles={'background-color': '#F6F6F6'}))]
        for doc in self.db_response:
            rlist.append(pn.Row(pn.pane.Str(doc)))
        return pn.WidgetBox(*rlist, width=600, scroll=True)

    @param.depends('convchain', 'clr_history')
    def get_chats(self):
        if not self.chat_history:
            return pn.WidgetBox(pn.Row(pn.pane.Str("No History Yet")), width=600, scroll=True)
        rlist = [pn.Row(pn.pane.Markdown(f"Current Chat History variable", styles={'background-color': '#F6F6F6'}))]
        for exchange in self.chat_history:
            rlist.append(pn.Row(pn.pane.Str(exchange)))
        return pn.WidgetBox(*rlist, width=600, scroll=True)

    def clr_history(self, event=None):
        self.chat_history = []
        self.panels = []
        self.answer = ""
        self.db_query = ""
        self.db_response = []



cb = cbfs()

file_input = pn.widgets.FileInput(accept='.pdf')
button_load = pn.widgets.Button(name="Load DB", button_type='primary')
button_clearhistory = pn.widgets.Button(name="Clear History", button_type='warning')
button_clearhistory.on_click(cb.clr_history)
inp = pn.widgets.TextInput(placeholder='Enter text here…')

bound_button_load = pn.bind(cb.call_load_db, button_load.param.clicks)
conversation = pn.bind(cb.convchain, inp.param.value)

jpg_pane = pn.pane.Image('./img/convchain.jpg')

tab1 = pn.Column(
    pn.Row(inp),
    pn.layout.Divider(),
    pn.panel(conversation, loading_indicator=True, height=300),
    pn.layout.Divider(),
    css_classes=['tab-content']
)
tab2 = pn.Column(
    pn.panel(cb.get_lquest),
    pn.layout.Divider(),
    pn.panel(cb.get_sources),
    css_classes=['tab-content']
)
tab3 = pn.Column(
    pn.panel(cb.get_chats),
    pn.layout.Divider(),
    css_classes=['tab-content']
)
tab4 = pn.Column(
    pn.Row(file_input, button_load, bound_button_load),
    pn.Row(button_clearhistory, pn.pane.Markdown("Clears chat history. Can use to start a new topic")),
    pn.layout.Divider(),
    pn.Row(jpg_pane.clone(width=400)),
    css_classes=['tab-content']
)
dashboard = pn.Column(
    pn.Row(pn.pane.Markdown('# YourOwnData_Bot', styles={'font-size': '32px', 'font-weight': 'bold', 'color': '#4CAF50'})),
    pn.Tabs(('Conversation', tab1), ('Database', tab2), ('Chat History', tab3), ('Configure', tab4)),
    css_classes=['dashboard']
)

pn.extension('css')
pn.config.raw_css.append('''
    .dashboard {
        background-color: #f0f0f0;
        padding: 20px;
        font-family: 'Arial', sans-serif;
    }
    .tab-content {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    .bk-btn-primary {
        background-color: #4CAF50;
        border-color: #4CAF50;
    }
    .bk-btn-warning {
        background-color: #FF9800;
        border-color: #FF9800;
    }
    .pn-Row {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
    }
    .pn-Column {
        display: flex;
        flex-direction: column;
        gap: 10px;
    }
    .pn-WidgetBox {
        display: flex;
        flex-direction: column;
        gap: 10px;
    }
    .pn-pane-Markdown {
        word-wrap: break-word;
    }
''')

#pn.serve(dashboard)
dashboard.servable()
