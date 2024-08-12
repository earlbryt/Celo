import getpass
from langchain.chains import create_history_aware_retriever, create_retrieval_chain       # retrievers
from langchain.chains.combine_documents import create_stuff_documents_chain   # for combining documents
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder     # prompt templates and chat history
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_message_histories import ChatMessageHistory   # stores chat history like the chat_history = []
from langchain_core.chat_history import BaseChatMessageHistory  # injects chat history into inputs and auto updates chat history after each invocation
from langchain_core.runnables.history import RunnableWithMessageHistory     # manages chat history (loads chat history from current session, runs rag_chain and saves the output)
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

import os
from dotenv import load_dotenv
import google.generativeai as genai


load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
llm = genai.GenerativeModel('gemini-1.5-flash')



def create_rag_system():
    docs = [
    {'document1': """ What we call happiness, in the strictest sense of the word,
    arises from the fairly sudden satisfaction of pent-up needs. 
    By its very nature it can be no more than an episodic phenomenon.

    hello there! You've set before me immense power over you! I see you, I
    know you - desire!

    Mans entire existence is summed up into one- the pursuit of pleasure
    and the avoidane of pain - from oneself, from the external world and
    from our relations with others

    Happiness, however, is something altogether subjective.There is no
    objective path to happiness: religion, the delusion of eternal life,
    fame,power, wealth, intellect, etc. Hapiness, nothing! is objective:
    Hapiness, EVERYTHING! is a purely subjective phenomenon. What he
    wants out of life is not what will make us happy!
    These are ideals, just ideals. In a perfect world, this is what we
    would want.

    ‘Thus conscience doth make cowards of us all …’ That a modern upbringing conceals from
    the young person the role that sexuality will play in his life is not the only criticism that
    must be levelled against it. Another of its sins is that it does not prepare him for the
    aggression of which he is destined to be the object. To send the young out into life with
    such a false psychological orientation is like equipping people who are setting out on a polar
    expedition with summer clothes and maps of the North Italian lakes. This reveals a certain
    misuse of ethical demands. The severity of these would do little harm if the educators said,
    ‘This is how people ought to be if they are to be happy and make others happy, but one
    must reckon with their not being like this.’ Instead, the young person is led to believe that
    everyone else complies with these ethical precepts and is therefore virtuous. This is the basis
    of the requirement that he too should become virtuous.


    If any man ever succeeds ALONE, without a grain of help from family,
    friends, teachers, religious and academic institutions, co-workers,
    then that man has and will never exist! 
    Not tech billionaires, not great performers, not the good student,
    not the recent startup, no one! has ever and will ever succeed 
    alone. And in the lot realized, each man receives according to his
    work!

    """},
    ]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    all_documents = []

    for doc in all_documents:
      for doc_name, content in doc.items():
        chunks = text_splitter.split_text(content)
        
        # Create Document objects for each chunk
        for chunk in chunks:
            all_documents.append(Document(
                page_content=chunk,
                metadata={"source": doc_name}
            ))
            
    vector_emb_store = Chroma.from_documents(
        documents=all_documents,
        embedding=embeddings
    )   

    retriever = vector_emb_store.as_retriever()
    
    # Context reconstruction prompt
    contextualized_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )

    contextualized_prompt_template = ChatPromptTemplate.from_messages([
        ('system', contextualized_system_prompt),
        MessagesPlaceholder(variable_name='chat_history'),
        ('human', '{question}'),
    ])

    # Create history-aware retriever
    history_aware_retriever = create_history_aware_retriever(
        llm,
        retriever,
        contextualized_prompt_template,
    )

    # RAG prompt
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Think through step by step"
        "\n\n"
        "{context}"
    )

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])

    # Create the RAG chain
    rag_chain = (
        RunnablePassthrough.assign(
            context=history_aware_retriever
        )
        | prompt_template
        | llm
        | StrOutputParser()
    )

    return rag_chain

# Create the Runnable RAG system
rag_system = create_rag_system()



# Chat history store
stores = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:         # store and load messages 
    if session_id not in stores:
        stores[session_id] = ChatMessageHistory()
    return stores[session_id]


conversational_rag_chain = RunnableWithMessageHistory(                      # LangChain object that manages chat history automatically                     
      rag_system,
      get_session_history,                                                       
      input_messages_key="input",
      history_messages_key="chat_history",
      output_messages_key="answer",
  )


def chatbot(user_input, session_id):                                        # ChatBot object      
    response = conversational_rag_chain.invoke(
        {input:user_input},
        config={"configurable": {"session_id": session_id}}  
    )
    return response['answer']                                               # We can input, chat_history and context as well


response = chatbot('What s space travel?', session_id="1")
print(response)







   