from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from kg_library.utils import Preprocessing
import json
from kg_library.utils import PathManager
import os

class TripletExtractor:
    def __init__(self, model_name="Babelscape/mrebel-large", src_lang="en_XX", tgt_lang="tp_XX"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang=src_lang, tgt_lang=tgt_lang, cache_dir=PathManager.get_mrebel_cache_path())
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir=PathManager.get_mrebel_cache_path())
        self.src_lang_token_id = self.tokenizer.convert_tokens_to_ids(src_lang)
        self.tgt_lang_token_id = self.tokenizer.convert_tokens_to_ids(tgt_lang)
        self.triplets = set()
        self.preprocessor = Preprocessing()
        self.type_map = {
            "per": "person",
            "misc": "character",
            "org": "organization",
            "loc": "location",
            "date": "date",
            "time": "time",
            "num": "number",
            "eve": "event",
            "cel": "cultural",
            "media": "title",
            "dis": "disease",
            "concept": "concept",
            "unk": "default"
        }

    def _extract_triplets_typed(self, text_element: str) -> list[dict]:
        triplets = []
        current = 'x'
        subject = relation = object_ = object_type = subject_type = ''

        for token in text_element.replace("<s>", "").replace("<pad>", "").replace("</s>", "").replace("tp_XX", "").replace(
                "__en__", "").split():
            if token == "<triplet>" or token == "<relation>":
                current = 't'
                if relation:
                    triplets.append({
                        'head': subject.strip(), 'head_type': self.type_map[subject_type],
                        'type': relation.strip(), 'tail': object_.strip(), 'tail_type': self.type_map[object_type]
                    })
                    relation = ''
                subject = ''
            elif token.startswith("<") and token.endswith(">"):
                if current in ['t', 'o']:
                    current = 's'
                    if relation:
                        triplets.append({
                            'head': subject.strip(), 'head_type': self.type_map[subject_type],
                            'type': relation.strip(), 'tail': object_.strip(), 'tail_type': self.type_map[object_type]
                        })
                    object_ = ''
                    subject_type = token[1:-1]
                else:
                    current = 'o'
                    object_type = token[1:-1]
                    relation = ''
            else:
                if current == 't':
                    subject += ' ' + token
                elif current == 's':
                    object_ += ' ' + token
                elif current == 'o':
                    relation += ' ' + token

        if all([subject, relation, object_, object_type, subject_type]):
            triplets.append({
                'head': subject.strip(), 'head_type': self.type_map[subject_type],
                'type': relation.strip(), 'tail': object_.strip(), 'tail_type': self.type_map[object_type]
            })

        return triplets

    def extract_from_text(self, text_element: str, num_beams=1, max_length=1024):
        model_inputs = self.tokenizer(text_element, max_length=max_length, padding=True, truncation=True, return_tensors="pt")
        generated_tokens = self.model.generate(
            input_ids=model_inputs["input_ids"].to(self.model.device),
            attention_mask=model_inputs["attention_mask"].to(self.model.device),
            decoder_start_token_id=self.tgt_lang_token_id,
            max_length=max_length,
            num_beams=num_beams,
            num_return_sequences=num_beams,
            length_penalty=0
        )

        decoded_preds = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=False)
        for sentence in decoded_preds:
            triplets = self._extract_triplets_typed(sentence)
            for t in triplets:
                self.triplets.add((t['head'], t['type'], t['tail'], t['head_type'], t['tail_type']))  # для GNN

    def extract_from_full_text(self, full_text: str, num_beans = 1) -> set:
        preprocessed_sentences = self.preprocessor.preprocessing(full_text)
        for sentence in preprocessed_sentences:
            self.extract_from_text(sentence, num_beams=num_beans)
        return self.triplets

    def print_knowledge_graph(self):
        print("Extracted Knowledge Graph:")
        for head, relation, tail, head_type, tail_type in sorted(self.triplets):
            print(f"  ({head} - [{head_type}]) -[{relation}]-> ({tail} - [{tail_type}])")

    def save_triplets_to_json(self, file_name="triplets.json"):
        PathManager.ensure_dirs()
        file_path = PathManager.get_output_path(file_name)

        with open(file_path, "w") as f:
            json.dump(list(self.triplets), f)

    def load_triplets_from_json(self, file_name="triplets.json"):
        output_path = PathManager.get_output_path(file_name)
        input_path = PathManager.get_input_path(file_name)

        if os.path.exists(output_path):
            file_path = output_path
        elif os.path.exists(input_path):
            file_path = input_path
        else:
            file_path = file_name

        with open(file_path, "r") as f:
            self.triplets = {tuple(inner_list) for inner_list in json.load(f)}