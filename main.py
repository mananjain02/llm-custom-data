import os
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import FAISS
import color_print
from constants import OPENAI_API_KEY
os.environ['OPENAI_API_KEY']=OPENAI_API_KEY


def main():
    try: 
        # check whether index has been already created
        if not os.path.isdir("pdf-index"):
            color_print.print_red("Index some data to get started")
            print("""
            You can add the pdfs you want to index in docs folder,
            then execute the index.py file to index the pdf into vector database
            """)
            return
        
        # set the length of history
        max_history_len = 3
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.load_local("pdf-index", embeddings=embeddings)
        llm = ChatOpenAI()
        history = []
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            chain_type='stuff',
            retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
            return_source_documents=True
        )
        while(1):
            query = color_print.print_green("Enter Your query")
            query = input()
            resp = chain({"question": query, "chat_history": history})
            history.append((query, resp["answer"]))
            while len(history)>max_history_len:
                history.pop(0)

            color_print.print_green("Answer")
            print(resp["answer"])

            color_print.print_green("Source documents")
            print(resp['source_documents'], "\n\n")
    except Exception as e:
        color_print.print_red(e)

if __name__=="__main__":
    color_print.print_green("LLMs on Custom Data")
    main()
    

