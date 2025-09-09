import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass(frozen=True)
class cfig:
    AZURE_SERVICE: str=os.getenv("AZURE_SEARCH_SERVICE","").strip()
    AZ_KEY:str=os.getenv("AZURE_SEARCH_KEY","").strip()
    AZ_INDEX:str=os.getenv("AZURE_SEARCH_INDEX","").strip()
    EMB_MODEL:str=os.getenv("EMB_MODEL","intfloat/e5-base-v2").strip()
    FT_MODEL: str = os.getenv("FT_MODEL_PATH").strip()
    google:str=os.getenv("google").strip()
    K_VECTOR:int=12
    K_LEXICAL: int =50
    FINAL_CONTEXT_K: int=8
    MAX_PER_SOURCE: int=2
    WEIGHT_VECTOR: float=0.6
    WEIGHT_LEXICAL: float=0.4   

def require_cfg(c:cfig)-> None:
    missing=[]
    if not c.AZ_SERVICE: missing.append("AZURE_SEARCH_SERVICE")
    if not c.AZ_KEY:     missing.append("AZURE_SEARCH_KEY")
    if not c.AZ_INDEX:   missing.append("AZURE_SEARCH_INDEX")
    if missing:
                raise RuntimeError(
            f"Missing required environment variables: {', '.join(missing)}\n"
            "Please set them in your .env file."
        )
    

