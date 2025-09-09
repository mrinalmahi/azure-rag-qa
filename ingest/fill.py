import os
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import uuid
load_dotenv()
key=os.getenv("AZURE_SEARCH_KEY")
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.normpath(os.path.join(script_dir, os.pardir))
loc=os.path.join(root_dir, "data", "raw")
service=os.getenv("AZURE_SEARCH_SERVICE")
index_name=os.getenv("AZURE_SEARCH_INDEX")
#connecting to client
client=SearchClient(
     endpoint=f"https://{service}.search.windows.net",
     index_name=index_name,
     credential=AzureKeyCredential(key)
)
#embedding model
embedding= SentenceTransformer("intfloat/e5-base-v2")
existing=[doc["source"] for doc in client.search("*",select=["source"])]
docs=[]
text_splitter=RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=50,
        length_function=len
    )

for name in os.listdir(loc):
    filepath=os.path.join(loc,name)
    if name in existing:
        continue
    loader=PyPDFLoader(filepath)
    document=loader.load()

    content=text_splitter.split_documents(document)
    vectors = embedding.encode([doc.page_content for doc in content], batch_size=16).tolist()
    for i,doc in enumerate(content):
        docs.append({
            "id":str(uuid.uuid4()),
            "content":doc.page_content,
            "source":name,
            "page":doc.metadata.get("page"),
            "contentVector":vectors[i]
        })
client.upload_documents(docs)
print(f"successfully uploaded {len(docs)} documents to {index_name}")