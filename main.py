from fastapi import FastAPI
from pydantic import BaseModel
from gramformer import Gramformer
import asyncio
import torch
import errant
import spacy
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins="*",  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

device = "cuda" if torch.cuda.is_available() else "cpu"

# Custom Gramformer to use en_core_web_md
class CustomGramformer(Gramformer):
    def __init__(self, models=1):
        super().__init__(models=models)
        nlp = spacy.load("en_core_web_md", disable=["ner"])  # Disable NER for speed
        self.annotator = errant.load("en", nlp=nlp)  # Use this model in errant

gf = CustomGramformer(models=1)

class TextInput(BaseModel):
    text: str

@app.post("/fix-grammar")
async def fix_grammar(data: TextInput):
    text = data.text

    # Split input into paragraphs (splitting on double newlines)
    paragraphs = text.split('\n\n')

    corrected_paragraphs = []

    for para in paragraphs:
        # Run correction in thread to avoid blocking
        corrected_list = await asyncio.get_event_loop().run_in_executor(
            None, lambda: list(gf.correct(para, max_candidates=1))
        )
        corrected_para = corrected_list[0] if corrected_list else para
        corrected_paragraphs.append(corrected_para)

    # Join corrected paragraphs back with double newline to keep paragraph breaks
    corrected = '\n\n'.join(corrected_paragraphs)

    return {"corrected_text": corrected}