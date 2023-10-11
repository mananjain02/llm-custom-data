import os
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
import color_print
from constants import OPENAI_API_KEY
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY


def get_pdf_paths(folder_path):
    """Returns a list of the paths of all the PDFs in the given folder."""
    pdf_paths = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".pdf"):
                pdf_paths.append(os.path.join(root, file))
    return pdf_paths

def index():
    try:
        """Read pdfs from docs folder and index them in vector database"""
        print("starting indexing...")
        folder_path = "docs/"
        text = []
        for path in get_pdf_paths(folder_path):
            print(path)
            loader = PyPDFLoader(path)
            text.extend(loader.load())
            print(f"Read data of file: {path}")
            os.remove(path=path)
        print("Creating text chunks...")
        text_splitter = CharacterTextSplitter(separator="\n", chunk_size=400, chunk_overlap=50, length_function=len)
        text_chunks = text_splitter.split_documents(text)
        print("creating embeddings")
        embeddings = OpenAIEmbeddings()
        if os.path.isdir("pdf-index"):
            print('pdf-index exists, loading')
            vectorstore = FAISS.load_local("pdf-index", embeddings=embeddings)
            print("adding to pdf-index")
            vectorstore.add_documents(text_chunks)
            print("saving pdf-index")
            vectorstore.save_local("pdf-index")
        else:
            print("pdf-index does not exists, creating")
            vectorstore = FAISS.from_documents(text_chunks, embeddings)
            print("saving locally")
            vectorstore.save_local("pdf-index")
        print("Docs indexed successfully!!")
    except Exception as e:
        color_print.print_red(e)

if __name__=="__main__":
    index()
