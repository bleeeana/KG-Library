from typing import List
import spacy, coreferee, pysbd, re
from spacy.language import Doc
from kg_library.common import data_frame

class Preprocessing:
    def __init__(self, model_name = 'en_core_web_lg'):
        if not spacy.util.is_package(model_name):
            spacy.cli.download(model_name)
        self.nlp = spacy.load(model_name)
        try:
            self.nlp.add_pipe('coreferee')
        except ValueError:
            pass
        self.segmenter = pysbd.Segmenter(language='en', clean=False)

    def __cleaning(self, text: str) -> str:
        clean_text = text.replace("“", '"').replace("”", '"')
        clean_text = re.sub(r'\s+', ' ', text).strip()
        return clean_text

    def __coreferee_preprocessing(self, coref_doc: Doc) -> str:
        resolved_text = ""
        for token in coref_doc:
            repres = coref_doc._.coref_chains.resolve(token)
            print(coref_doc._.coref_chains)
            if repres:
                resolved_text += " " + " and ".join(
                    [t.text if t.ent_type_ == "" else [e.text for e in coref_doc.ents if t in e][0] for t in repres])
            else:
                resolved_text += " " + token.text

        return resolved_text

    def __set_mode(self):
        spacy.prefer_gpu()
        if spacy.prefer_gpu():
            print("Используется GPU")
        else:
            print("Используется CPU")

    def __split_text_to_segments(self, text: str) -> List[str]:
        return self.segmenter.segment(text)

    def preprocessing(self, text: str) -> str:

        resolved_text = self.__coreferee_preprocessing(self.nlp(text))
        print(resolved_text)
        #splitted_text = self.__split_text_to_segments(resolved_text)
        clean_text = re.sub(r'\s+([.,!?])', r'\1', resolved_text)
        return clean_text

    def __lemmatization(self, text: str) -> str:
        if not text or not isinstance(text, str):
            return ""
        lemmatized_text = ' '.join([token.lemma_ for token in self.nlp(text)])
        return lemmatized_text