from typing import List
import spacy, pysbd, re
from spacy.language import Doc

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

    @staticmethod
    def _coreferee_preprocessing(coref_doc: Doc) -> str:
        resolved_tokens = []
        for token in coref_doc:
            resolved_parts_list = coref_doc._.coref_chains.resolve(token)
            if resolved_parts_list:
                resolved_parts = []
                for t in resolved_parts_list:
                    if t.ent_type_ == "":
                        resolved_parts.append(t.text)
                    else:
                        matching_entities = [e.text for e in coref_doc.ents if t in e]
                        if matching_entities:
                            resolved_parts.append(matching_entities[0])
                        else:
                            resolved_parts.append(t.text)
                resolved_tokens.append(" and ".join(resolved_parts))
            else:
                resolved_tokens.append(token.text)

        return " ".join(resolved_tokens)

    def __split_text_to_segments(self, text: str) -> List[str]:
        return self.segmenter.segment(text)

    def preprocessing(self, text: str) -> List[str]:
        resolved_text = self._coreferee_preprocessing(self.nlp(text))
        print(resolved_text)
        clean_text = re.sub(r'\s+([.,!?])', r'\1', resolved_text)
        split_text = self.__split_text_to_segments(clean_text)
        return split_text