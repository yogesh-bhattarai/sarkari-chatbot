import os
import textwrap
import asyncio
from pathlib import Path

#from google.colab import userdata
from IPython.display import Markdown
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Qdrant
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from llama_parse import LlamaParse
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain.schema import Document  # Correct import for Document


os.environ["GROQ_API_KEY"] =("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")


def print_response(response):
    response_txt = response["result"]
    for chunk in response_txt.split("\n"):
        if not chunk:
            print()
            continue
        print("\n".join(textwrap.wrap(chunk, 100, break_long_words=False)))


async def load_document(file_path):
    llama_parse_documents = await parser.aload_data(file_path)
    return llama_parse_documents
file_path = "./data/sarkari chatbot.pdf"



import os
from llama_parse import LlamaParse
from IPython.display import Markdown
import nest_asyncio
import asyncio

# Allow nested asyncio loops (for Jupyter and similar environments)
nest_asyncio.apply()

# Set the API key
os.environ["LLAMA_PARSE"] = "llx-8k3jeCf570YjQQLz5qaMXinq8ZJgVOugf5C4397sAnn6UJxd"

# Instruction for parsing
instruction = """This PDF is about rules for getting citizenship in Nepal.
It gives detailed information and cases for citizenship. 
Try to be very precise while answering the questions, give answers in very short."""

# Initialize the parser
parser = LlamaParse(
    api_key="llx-8k3jeCf570YjQQLz5qaMXinq8ZJgVOugf5C4397sAnn6UJxd",
    result_type="markdown",
    parsing_instruction=instruction,
    max_timeout=5000,
)

# Asynchronous function for parsing
async def parse_document(file_path):
    print("Parsing document...")
    llama_parse_documents = await parser.aload_data(file_path)
    print("Parsed documents successfully!")
    return llama_parse_documents

# Main code execution
file_path = "./data/sarkari chatbot.pdf"  # Replace with the actual path to your file

# Use asyncio to run the parse_document function
parsed_documents = asyncio.run(parse_document(file_path))

# Access the first parsed document
if parsed_documents:
    parsed_doc = parsed_documents[0]  # Get the first parsed document
    print("First document text preview:")
    #Markdown(parsed_doc.text[:4096])  # Render in Jupyter Notebook if needed
else:
    print("No documents were parsed.")



# Define a function to safely read the file
def safe_read_file(file_path, encoding="utf-8", fallback_encoding="windows-1252"):
    try:
        with open(file_path, "r", encoding=encoding) as file:
            return file.read()
    except UnicodeDecodeError:
        # Retry with fallback encoding
        with open(file_path, "r", encoding=fallback_encoding) as file:
            return file.read()

# Load the file with encoding handled
file_path = "data/parsed_document.md"
file_content = safe_read_file(file_path)

# Create a custom loader to process the content
class CustomMarkdownLoader(UnstructuredMarkdownLoader):
    def _get_elements(self):
        # Directly use the preloaded content
        return [Document(page_content=file_content)]

# Use the custom loader
loader = CustomMarkdownLoader(file_path)
loaded_documents = loader.load()

# Check loaded content
# for doc in loaded_documents:
#     print(f"Content: {doc.page_content}\nMetadata: {doc.metadata}")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=2048, chunk_overlap=128)
docs = text_splitter.split_documents(loaded_documents)
len(docs)

from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
qdrant = Qdrant.from_documents(
    docs,
    embeddings,
    #location=":memory:",
    path="./db",
    collection_name="document_embeddings",
)
query = "Can a local resident who is permanently settled get citizenship?"
similar_docs = qdrant.similarity_search_with_score(query)

# for doc, score in similar_docs:
#
#     print(f"text: {doc.page_content[:256]}\n")
#     print(f"score: {score}")
#     print("-" * 80)
#     print()


# Qdrant ko retriever banaune with search parameters
retriever = qdrant.as_retriever(search_kwargs={"k": 5})  # Top 5 similar documents retrieve garne

# Query ko adhar ma documents retrieve garne
retrieved_docs = retriever.invoke(query)


compressor = FlashrankRerank(model="ms-marco-MiniLM-L-12-v2")
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=retriever
)


#%%

llm = ChatGroq(temperature=0, model_name="llama3-70b-8192")


prompt_template = """
Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Answer the question with the short reason and dont provide additional helpful information, and also dont give source documents
just give short answer.

Responses should be properly formatted to be easily read.
"""

# PromptTemplate ko instance create gareko cha
prompt = PromptTemplate(
    template=prompt_template,  # Template ko string
    input_variables=["context", "question"]  # Input variables specify garne
)


qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=compression_retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt, "verbose": True},
)


response = qa.invoke("To give Nepali citizenship certificate to citizens based on what documents?")


qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=compression_retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt, "verbose": False},
)

response = qa.invoke("i want to make citizenship?")


print_response(response)

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load the M2M100 model and tokenizer
model_name = "facebook/m2m100_418M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)


def translate_to_nepali(text):
    """
    Translate the given text from English to Nepali using the M2M100 model.
    """
    tokenizer.src_lang = "en"  # Source language: English
    inputs = tokenizer(text, return_tensors="pt")

    # Generate translation to Nepali
    outputs = model.generate(inputs["input_ids"], forced_bos_token_id=tokenizer.lang_code_to_id["ne"])
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return translated_text


# Test the translation function
if __name__ == "__main__":
    english_text = "What are the required documents for Nepali citizenship?"
    nepali_translation = translate_to_nepali(english_text)
    print("Original Text:", english_text)
    print("Translated Text:", nepali_translation)



