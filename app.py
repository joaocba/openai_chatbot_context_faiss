##
# Uses LangChain and FAISS as vector store to index data (also index the data localy for reuse - folder faiss_index)
# Uses SpacyTextSplitter to split text into sentences by each \n
# Support multiple files to be vectorized (folder)

## Documentation:
# Docs about Faiss: https://langchain.readthedocs.io/en/latest/modules/indexes/vectorstore_examples/faiss.html
# Docs about embeddings: https://langchain.readthedocs.io/en/latest/reference/modules/embeddings.html
# Docs about TextSplitter: https://langchain.readthedocs.io/en/latest/modules/indexes/examples/textsplitter.html

## Dependecies:
# pip install flask
# pip install langchain
# nltk.download('averaged_perceptron_tagger')
# pip install unstructured
# pip install python-magic-bin
# pip install faiss-cpu
# pip install spacy
# python -m spacy download en


from flask import Flask, render_template, request
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter, SpacyTextSplitter
from langchain import OpenAI, VectorDBQA
from langchain.document_loaders import DirectoryLoader
import magic
import nltk
import os
import logging
import datetime


# create logs folder if it doesn't exist
if not os.path.exists("logs"):
    os.makedirs("logs")

# get current date and time for log filename
log_filename = f"logs/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"

# configure logger
logging.basicConfig(filename=log_filename, level=logging.INFO)


# set OpenAI API
# os.environ["OPENAI_API_KEY"] = ''
# in case it is already defined on windows path variables
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")


# set directory
loader = DirectoryLoader('./content/empresas', glob='**/*.txt')
documents = loader.load()

# settings for text
#text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
text_splitter = SpacyTextSplitter(chunk_size=1000, chunk_overlap=0) # better splitter
texts = text_splitter.split_documents(documents)

# create embeddings
embeddings = OpenAIEmbeddings(openai_api_key=os.environ['OPENAI_API_KEY'])

# search texts on docs with embeddings
#docsearch = FAISS.from_documents(texts, embeddings)

# save on local to avoid recreation
docsearch = FAISS.from_documents(texts, embeddings)
docsearch.save_local("faiss_index")
new_docsearch = FAISS.load_local("faiss_index", embeddings)
#docs = new_db.similarity_search(query)


# array to store conversations
conversation = ["You are a virtual assistant and you speak portuguese."]    # define initial role

app = Flask(__name__)

# define app routes
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():

    # index data
    #qa = VectorDBQA.from_chain_type(llm=OpenAI(max_tokens = 150), chain_type="stuff", vectorstore=docsearch)
    qa = VectorDBQA.from_chain_type(llm=OpenAI(max_tokens = 600), chain_type="stuff", vectorstore=new_docsearch)

    # get user input
    user_input = request.args.get("msg") + '\n'
    response = ''
    if user_input:
        conversation.append(f"{user_input}")

        # get conversation history
        prompt = "\n".join(conversation[-3:])

        # generate AI response based on indexed data
        response = qa(prompt)
        #print(response)

        # add AI response to conversation
        conversation.append(f"{response}")

        # log conversation
        with open(log_filename, "a") as f:
            f.write(f"User: {user_input}\n")
            f.write(f"AI: {response}\n\n")

        # log conversation using logger
        logging.info(f"User: {user_input}")
        logging.info(f"AI: {response}")

    return response['result'] if response else "Sorry, I didn't understand that."


if __name__ == "__main__":
    app.run()