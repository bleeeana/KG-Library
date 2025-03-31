from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from kg_library.common import data_frame
from kg_library.utils import preprocessing

class KnowledgeGraphExtractor:
    def __init__(self, model_name='Babelscape/rebel-large'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.knowledge_graph = set()  # Используем множество для фильтрации повторов

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

    def extract_from_text(self, text, dynamic_sequences=True, base_sequences=3):
        model_inputs = self.tokenizer(text, max_length=1024, padding=True, truncation=True, return_tensors='pt')
        num_tokens = len(model_inputs['input_ids'][0])
        print(f"Number of tokens: {num_tokens}")
        num_sequences = base_sequences if not dynamic_sequences else max(2, min(10, num_tokens))
        print(f"Number of sequences: {num_sequences}")
        gen_kwargs = {
            "max_length": 2048,
            "length_penalty": 0,
            "num_beams": num_sequences,
            "num_return_sequences": num_sequences
        }
        generated_tokens = self.model.generate(**model_inputs, **gen_kwargs)
        decoded_preds = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=False)
        print("Decoded predictions:" + str(decoded_preds))
        print(generated_tokens)
        for sentence_pred in decoded_preds:
            print(sentence_pred)
            self.knowledge_graph.update(self.__extract_relations(sentence_pred))

    def print_knowledge_graph(self):
        print("Extracted Knowledge Graph:")
        for head, relation, tail in sorted(self.knowledge_graph):
            print(f"  ({head}) -[{relation}]-> ({tail})")

if __name__ == "__main__":
    text = "Rowling wrote Harry Potter. She wrote it when she was 16 years old. The book was published by Bloomsbury."
    lemmatized_text = preprocessing(text)
    print(lemmatized_text)
    extractor = KnowledgeGraphExtractor()
    extractor.extract_from_text(lemmatized_text)
    extractor.print_knowledge_graph()