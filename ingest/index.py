import os
import sys
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexClient 
from azure.search.documents.indexes.models import (
    SearchIndex, SimpleField, SearchableField, SearchField,
    SearchFieldDataType, VectorSearch, VectorSearchAlgorithmConfiguration,
    HnswParameters, VectorSearchProfile,HnswAlgorithmConfiguration
)
load_dotenv()
key=os.getenv("AZURE_SEARCH_KEY")
index_name=os.getenv("AZURE_SEARCH_INDEX")
service=os.getenv("AZURE_SEARCH_SERVICE")
#connecting to client
index_client=SearchIndexClient (
    endpoint=f"https://{service}.search.windows.net",
    credential=AzureKeyCredential(key)
)
existing=[idx.name for idx in index_client.list_indexes()]
if index_name in existing:
    choice=input("index '{index_name}' already existists, Do you want to overwrite Y/N")
    if(choice.lower()=='y'):
        index_client.delete_index(index_name)
    else:
        sys.exit()

index=SearchIndex(
    name=index_name,
    fields=[
        SimpleField(name="id",type=SearchFieldDataType.String,key=True),    
        SearchableField(name="content",type=SearchFieldDataType.String),
        SimpleField(name="source",type=SearchFieldDataType.String,filterable=True),
        SimpleField(name="page",type=SearchFieldDataType.Int32,filterable=True),
        SearchField(
    name="contentVector",
    type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
     vector_search_dimensions=768,
     vector_search_profile_name="vprof"
     ),],
    vector_search=VectorSearch(
        algorithms=
        [            HnswAlgorithmConfiguration(
                name="hnsw",
                parameters=HnswParameters(m=16, ef_construction=200),
            )
        
             ],
             profiles=[VectorSearchProfile(name="vprof", algorithm_configuration_name="hnsw")]
)
)
index_client.create_index(index)
print("index created:{index_name}")


