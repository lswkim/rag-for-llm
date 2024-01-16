import streamlit as st
import os

# Langchain
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms.bedrock import Bedrock
from langchain.chains import RetrievalQA, ConversationChain
from langchain.callbacks.base import BaseCallbackHandler

# aws
import boto3

# custom
import config

class StreamHandler(BaseCallbackHandler):
    """
        Streaming output class
    """
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)

# 기본 UI 세팅
st.title("Chatbot Demo")

def get_index_list():
    st.session_state["index_options"] = os.listdir(st.session_state['index_path'])

# init page
def on_start():
    """ Default setting that does not have to start at refresh
    """
    # set vectorDB path
    st.session_state['index_changed'] = False
    st.session_state['index_path'] = './indexs/'
    # Load Embedding Model
    embed_model_nm = 'jhgan/ko-sroberta-multitask'
    model_kwargs = {'device':'cpu'}
    encode_kwargs = {'normalize_embeddings': False}

    st.session_state["embeddings"] = HuggingFaceEmbeddings(
        model_name=embed_model_nm,     # Provide the pre-trained model's path
        model_kwargs=model_kwargs, # Pass the model configuration options
        encode_kwargs=encode_kwargs # Pass the encoding options
    )

    # Load Index list from path
    get_index_list()

if "on_start" not in st.session_state:
    st.session_state["on_start"] = True
    on_start()

if "retrival_model" not in st.session_state:
    st.session_state["retrival_model"] = ""

# change retrival model
def update_retrival_model():
    if (st.session_state["retrival_model"] == "") \
        or (st.session_state["retrival_model"] is None):
        return
 
    if not st.session_state['index_changed']:
        st.session_state['index_changed'] = True
    else:
        print('retrieval setting')
        st.session_state['index_changed'] = False
        st.session_state['db'] = FAISS.load_local(f'{st.session_state["index_path"]}{st.session_state["retrival_model"]}'
                                                , st.session_state["embeddings"])
        st.session_state['retrieval'] = st.session_state['db'].as_retriever(search_kwargs={"k": 4})

st.session_state["retrival_model"] = st.sidebar.selectbox(
    label="Choose Documents"
    , index=None
    , options=st.session_state["index_options"]
    , placeholder="Choose an option"
    , on_change=update_retrival_model()
)

if st.session_state['index_changed']: update_retrival_model()

# llm parameter
st.session_state['temperature'] = st.sidebar.number_input(label="temperature", min_value=0.0, max_value=1.0, value=1.0) # 낮은 값을 사용하면 반응의 임의성을 줄일 수 있습니다. (default: 1)
st.session_state['top-p'] = st.sidebar.number_input(label="top-p", min_value=0.0, max_value=1.0, value=0.99) # 가능성이 낮은 옵션을 무시하려면 낮은 값을 사용합니다. (default: 0.999)
st.session_state['top-k'] = st.sidebar.number_input(label="top-k", min_value=0, value=250) # 모델이 다음 토큰을 생성하는 데 사용하는 토큰 선택 개수를 지정합니다.(default: 250)

# message container
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask something..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        session = boto3.Session(region_name=config.region_name)
        bedrock_runtime = session.client(service_name="bedrock-runtime"
                                         , aws_access_key_id=config.aws_access_key_id
                                         , aws_secret_access_key=config.aws_secret_access_key)
        llm_model_nm = 'anthropic.claude-v2'
        model_kwargs = {
                        "max_tokens_to_sample": 2048,
                        "temperature":st.session_state['temperature'],
                        "top_p":st.session_state['top-p'],
                        "top_k":st.session_state['top-k'],
        }

        llm=Bedrock(
                    model_id=llm_model_nm,
                    client=bedrock_runtime,
                    model_kwargs=model_kwargs,
                    streaming=True,
                    callbacks=[StreamHandler(st.empty())]
                )

        # with out retrieval
        if st.session_state["retrival_model"] is None:
            conversation = ConversationChain(
                llm=llm
                # , memory=ConversationBufferMemory(ai_prefix="Assistant")
                # , verbose=True
            )
            result = conversation.predict(input=prompt)
            st.session_state.messages.append({"role": "assistant", "content": result})

        # with retrieval
        else:
            qa = RetrievalQA.from_chain_type(
                llm=llm
                , chain_type="stuff"
                , retriever=st.session_state['retrieval']
                , return_source_documents=True
                , input_key="question"
            )

            response = qa({"question": prompt}, return_only_outputs=True)

            for document in response['source_documents']:
                print(document.page_content)

            st.session_state.messages.append({"role": "assistant", "content": response['result']})

