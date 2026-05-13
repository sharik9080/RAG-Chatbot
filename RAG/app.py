import chainlit as cl
from rag import *
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOllama
from langchain.memory import ConversationBufferMemory
from langchain_ollama import ChatOllama

doc_search = load_and_retrieve_docs('pdf')
embeddings = OllamaEmbeddings(model = 'nomic-embed-text')

@cl.on_chat_start
async def on_chat_start():

    # Create a retriever that uses the Chroma vector store
    retriever = doc_search.as_retriever()

    message_history = ChatMessageHistory()

    memory = ConversationBufferMemory(
        memory_key = "chat_history",
        output_key = "answer",
        chat_memory = message_history,
        return_messages = True,
    )

    # Create a chain that uses the Chroma vector store
    chain = ConversationalRetrievalChain.from_llm(
        ChatOllama(model = "llama3.2"),
        chain_type = "stuff",
        retriever = retriever,
        memory = memory,
        return_source_documents = True,
    )

    cl.user_session.set("chain", chain)


@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")  # type: ConversationalRetrievalChain
    cb = cl.AsyncLangchainCallbackHandler()
    text_elements = []  

    # --- Step 1: Try retrieval with scores
    retrieved_docs = retrieve_with_scores(doc_search, message.content, k=3)
    best_score = retrieved_docs[0].metadata.get("score", 0) if retrieved_docs else 0
    
    # --- Step 2: Decide whether to use RAG or fallback
    SCORE_THRESHOLD = 0.4
    if best_score >= SCORE_THRESHOLD:
        # Use RAG
        res = await chain.ainvoke(message.content, callbacks=[cb])
        answer = res["answer"]
        source_documents = res["source_documents"]

        show_source = True #
        if show_source:
            if source_documents:
                for source_idx, source_doc in enumerate(source_documents):
                    source_name = f"source_{source_idx+1}"
                
                    text_elements.append(
                        cl.Text(content = f"Page Number: {source_doc.metadata.get('page')}\nSource: {source_doc.metadata.get('source')}",
                                name = source_name)
                    )   # also add ("\nContent:{source_doc.page_content}") to show relevent context
        else:
            answer += "\nTo show the sources change show_source = True in the app.py file"
    else:
        # Fallback → use base LLM without retrieval
        llm = ChatOllama(model="llama3.2")
        answer = (await llm.ainvoke(message.content)).content
        source_documents = []

    # res = await chain.ainvoke(message.content, callbacks=[cb])
    # answer = res["answer"]
    # source_documents = res["source_documents"]  



    await cl.Message(content = answer, elements = text_elements).send()

@cl.on_stop
def on_stop():
    print("The user wants to stop the task!")


@cl.on_chat_end
def on_chat_end():
    print("The user disconnected!")
 


