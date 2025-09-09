from typing import Tuple, List
from langchain.schema import Document
from config import cfig
import os
cfg=cfig()
os.environ["AZURE_AI_SEARCH_SERVICE_NAME"] = cfg.AZURE_SERVICE
os.environ["AZURE_AI_SEARCH_INDEX_NAME"]=cfg.AZ_INDEX
os.environ["AZURE_AI_SEARCH_API_KEY"]=cfg.AZ_KEY
os.environ["AZURESEARCH_FIELDS_CONTENT_VECTOR"] = "contentVector" 
from langchain_community.vectorstores.azuresearch import AzureSearch
from langchain_huggingface import HuggingFaceEmbeddings
azure_endpoint: str = f"https://{cfg.AZURE_SERVICE}.search.windows.net"
embedding=HuggingFaceEmbeddings(
    model_name=cfg.EMB_MODEL,
    encode_kwargs={"normalize_embeddings":True}
)

vectorstore = AzureSearch(
    azure_search_endpoint=f"https://{cfg.AZURE_SERVICE}.search.windows.net",
    azure_search_key=cfg.AZ_KEY,
    index_name=cfg.AZ_INDEX,
    embedding_function=embedding,
)

retriever = vectorstore.as_retriever(
    search_type="hybrid",

)

def pretty_source(name:str)->str:
    return name.rsplit(".",1)[0]if name else "source"

def cap_diversity(docs:List[Document],k:int,max_per_source:int) -> List[Document]:
    out,per=[],{}
    for d in docs:
        src=d.metadata.get("source","unknown")
        if per.get(src,0)>=max_per_source:
            continue
        per[src]=per.get(src,0)+1
        out.append(d)
        if(len(out)>=k):
            break
    return out 
def format_content(docs:List[Document])->str:
    lines=[]
    for d in docs:
        src=pretty_source(d.metadata.get("source"))
        page = d.metadata.get("page")
        lines.append(f"- {d.page_content}\n  [Reference: {src}, p.{page}]")
    return "\n\n".join(lines)
    
def retrieve_formatted(query: str) -> Tuple[List[Document], str]:
    q = "query: " + query
    raw_docs = retriever.invoke(q)
    kept = cap_diversity(raw_docs, k=cfg.FINAL_CONTEXT_K, max_per_source=cfg.MAX_PER_SOURCE)
    ctx = format_content(kept)
    return kept, ctx
