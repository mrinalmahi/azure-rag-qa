import os
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from peft import PeftModel
from config import cfig
BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

def load_llm(cfg:cfig):
    adapter_or_model_path = getattr(cfg, "FT_MODEL", None)
    print("Using path:", repr(adapter_or_model_path))
    tok=AutoTokenizer.from_pretrained(BASE_MODEL,use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base=AutoModelForCausalLM.from_pretrained(BASE_MODEL,device_map="auto")
    model = PeftModel.from_pretrained(base, adapter_or_model_path)
    model.eval()
    
    pipe=pipeline(
        "text-generation",
        model=model,
        tokenizer=tok,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
    )
    llm = HuggingFacePipeline(
        pipeline=pipe,
        pipeline_kwargs={
            "max_new_tokens": 90,       
            "do_sample": False,         
            "no_repeat_ngram_size": 4,
            "repetition_penalty": 1.15,
            "return_full_text": False,   
        },
    )
    llm = llm.bind(stop=["\n- ", "\n[Reference", "\nContext:", "\nQuestion:", "\nAnswer:"])
    return llm

def google_llm(cfg):
    model=ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    api_key=cfg.google  
)
    return model



RAG_PROMPT = PromptTemplate(
    input_variables=["question", "context"],
    template=(
        "You are an assistant for question-answering tasks with expertise in drug-target interaction. "
        "Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer, just say that you don't know. "
        "Use three sentences maximum and keep the answer concise and give the reference from the context.\n"
        "Question: {question}\n"
        "Context: {context}\n"
        "Answer:"
    ),
)


def build_rag_chain(cfg:cfig,choice):
    if choice==1:
        llm=google_llm(cfg)
    else:
        llm=load_llm(cfg)
    chain=RAG_PROMPT| llm |StrOutputParser()
    return chain
def generate_answer(cfg: cfig, question: str, context: str,choice:int) -> str:
    chain = build_rag_chain(cfg,choice)
    return chain.invoke({"question": question, "context": context})