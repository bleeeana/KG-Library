import spacy, coreferee
from kg_library.common import data_frame
import re

if not spacy.util.is_package("en_core_web_trf"):
    spacy.cli.download("en_core_web_trf")

nlp = spacy.load("en_core_web_trf")
try:
    nlp.add_pipe('coreferee')
except ValueError:
    pass

def preprocessing(text: str) -> str:
    if not text or not isinstance(text, str):
        return ""
    spacy.prefer_gpu()
    if spacy.prefer_gpu():
        print("Используется GPU")
    else:
        print("Используется CPU")
    coref_doc = nlp(text)
    resolved_text = ""
    for token in coref_doc:
        repres = coref_doc._.coref_chains.resolve(token)
        if repres:
            resolved_text += " " + " and ".join(
                [t.text if t.ent_type_ == "" else [e.text for e in coref_doc.ents if t in e][0] for t in repres])
        else:
            resolved_text += " " + token.text
    lemmatized_text = ' '.join([token.lemma_ for token in nlp(resolved_text)])
    clean_text = ' '.join(resolved_text.split())
    clean_text = re.sub(r'\s+([.,!?])', r'\1', clean_text)
    return clean_text