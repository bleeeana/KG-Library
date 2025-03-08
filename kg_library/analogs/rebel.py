from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from langchain_community.document_loaders import WikipediaLoader
'''
Этот код предназначен для извлечения связей из текста с помощью модели Babelscape/rebel-large.
Плюс решения: 1. Хорошо работает с небольшими текстами
Минусы решения: 1. Плохо работает с большими текстами, выдает повторяющиеся триплеты, неверные
                2. не формирует эмбеддинги, только текстовые триплеты
'''
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024,
    length_function=len,
    is_separator_regex=False, )
query = "Dune (Frank Herbert)"
raw_documents = WikipediaLoader(query=query).load_and_split(text_splitter=text_splitter)

tokenizer = AutoTokenizer.from_pretrained('Babelscape/rebel-large')
model = AutoModelForSeq2SeqLM.from_pretrained('Babelscape/rebel-large')

def extract_relations_from_model_output(text):
    relations = []
    subject, relation, object_ = '', '', ''
    text = text.strip()
    current = 'x'
    text_replaced = text.replace("<s>", "").replace("<pad>", "").replace("</s>", "")
    for token in text_replaced.split():
        if token == "<triplet>":
            current = 't'
            if relation != '':
                relations.append({
                    'head': subject.strip(),
                    'type': relation.strip(),
                    'tail': object_.strip()
                })
                relation = ''
                subject = ''
        elif token == "<subj>":
            current = 's'
            if relation != '':
                relations.append({
                    'head': subject.strip(),
                    'type': relation.strip(),
                    'tail': object_.strip()
                })
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
    if subject != '' and relation != '' and object_ != '':
        relations.append({
            'head': subject.strip(),
            'type': relation.strip(),
            'tail': object_.strip()
        })
    return relations


def are_relations_equal(r1, r2):
    return all(r1[attr] == r2[attr] for attr in ["head", "type", "tail"])


class KB():
    def __init__(self):
        self.relations = []

    def exists_relation(self, r1):
        return any(are_relations_equal(r1, r2) for r2 in self.relations)

    def add_relation(self, r):
        if not self.exists_relation(r):
            self.relations.append(r)

    def print(self):
        print("Relations:")
        for r in self.relations:
            print(f"  {r}")


def from_small_text_to_kb(text, verbose=False):
    kb = KB()
    model_inputs = tokenizer(text, max_length=1024, padding=True, truncation=True, return_tensors='pt')
    if verbose:
        print(f"Num tokens: {len(model_inputs['input_ids'][0])}")
    gen_kwargs = {
        "max_length": 512,
        "length_penalty": 0,
        "num_beams": 3,
        "num_return_sequences": 3
    }
    generated_tokens = model.generate(
        **model_inputs,
        **gen_kwargs,
    )
    decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=False)
    for sentence_pred in decoded_preds:
        relations = extract_relations_from_model_output(sentence_pred)
        for r in relations:
            print(r)
            if r["head"] and r["tail"] and r["type"]:  # Исключаем пустые связи
                kb.add_relation(r)

    return kb

def main():
    print(raw_documents)
    for doc in raw_documents:
        kb = from_small_text_to_kb(doc.page_content, verbose=True)
        for relation in kb.relations:
            head = relation['head']
            relationship = relation['type']
            tail = relation['tail']
            cypher = f"MERGE (h:`{head}`)" + f" MERGE (t:`{tail}`)" + f" MERGE (h)-[:`{relationship}`]->(t)"
            print(cypher)


if __name__ == "__main__":
    main()