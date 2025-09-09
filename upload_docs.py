import os
from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient


load_dotenv()
conn_name=os.getenv("AZURE_BLOB_CONTAINER")
conn_key=os.getenv("AZURE_BLOB_CONN")
client = BlobServiceClient.from_connection_string(conn_key)
container_client = client.get_container_client(conn_name)
existing_files=[b.name for b in container_client.list_blobs()]
for name in os.listdir("data/raw"):
    path=os.path.join("data/raw",name)
    if name in existing_files:
        continue
    try:
        with open(path,"rb") as f:
            container_client.upload_blob(name=name,data=f,overwrite=True)
            print("uploaded:",name)
    except Exception as e:
        print("Error uploading",e)
