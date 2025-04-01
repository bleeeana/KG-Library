from kg_library.common import data_frame
from kg_library.utils import Preprocessing
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class KnowledgeGraphExtractor:
    def __init__(self, model_name="Babelscape/rebel-large"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.triplets = set()
        self.preprocessor = Preprocessing()

    def __extract_relations(self, text):
        relations = set()
        subject, relation, object_ = '', '', ''
        current = 'x'
        text_replaced = text.replace("<s>", "").replace("<pad>", "").replace("</s>", "")

        for token in text_replaced.split():
            if token == "<triplet>":
                current = 't'
                if relation and (subject, relation, object_) not in relations:
                    relations.add((subject.strip(), relation.strip(), object_.strip()))
                relation, subject = '', ''
            elif token == "<subj>":
                current = 's'
                if relation and (subject, relation, object_) not in relations:
                    relations.add((subject.strip(), relation.strip(), object_.strip()))
                object_ = ''
            elif token == "<obj>":
                current = 'o'
                relation = ''
            else:
                if current == 't':
                    subject += ' ' + token
                elif current == 's':
                    object_ += ' ' + token
                elif current == 'o':
                    relation += ' ' + token
        if subject and relation and object_ and (subject, relation, object_) not in relations:
            relations.add((subject.strip(), relation.strip(), object_.strip()))
        return relations

    def extract_from_text(self, text : str, num_beams=2, max_length=512):
        text = self.preprocessor.preprocessing(text)
        print(text)
        model_inputs = self.tokenizer(text, max_length=max_length, padding=True, truncation=True, return_tensors="pt")
        print(model_inputs)
        gen_kwags = {
            "max_length": max_length,
            "num_beams": num_beams,
            "length_penalty": 0.0,
            "num_return_sequences": num_beams
        }
        generated_tokens = self.model.generate(
            **model_inputs,
            **gen_kwags
        )

        decoded_preds = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=False)
        print(decoded_preds)
        for sentence_pred in decoded_preds:
            self.triplets.update(self.__extract_relations(sentence_pred))

    def print_knowledge_graph(self):
        print("Extracted Knowledge Graph:")
        for head, relation, tail in sorted(self.triplets):
            print(f"  ({head}) -[{relation}]-> ({tail})")


if __name__ == "__main__":
    text = "Rowling wrote Harry Potter. It was published by Bloomsbury."
    extractor = KnowledgeGraphExtractor()
    extractor.extract_from_text(text)
    extractor.print_knowledge_graph()