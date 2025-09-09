from config import cfig
from retreivers import retrieve_formatted
from generator import generate_answer

def get_choice():
    while True:
        choice=int(input("press 1 for google response and 2 for local model"))
        if choice  in (1,2):
            return choice

def rag_answer(question: str):
    cfg = cfig()  
    docs, ctx = retrieve_formatted(question)
    choice=get_choice()
    print("working")
    answer=generate_answer(cfg,question,ctx,choice)
    print(answer)
    
if __name__ == "__main__":

    while True:
        q = input("Enter question:").strip()
        if(q.lower()=="exit"):
            break
        rag_answer(q)
         

    
    