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
            "per": "character",
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
            num_beams=max((len(text_element) // 300), num_beams),
            num_return_sequences=max((len(text_element) // 300), num_beams),
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
            print(sentence, end="\n\n")
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


def main():
    text = "The novel is told in epistolary format, as a series of letters, diary entries, ships' log entries, and so forth. The main writers of these items are also the novel's protagonists. The story is occasionally supplemented with newspaper clippings that relate events not directly witnessed by the story's characters. The tale begins with Jonathan Harker, a newly qualified English solicitor, journeying by train and carriage from England to Count Dracula's crumbling, remote castle (situated in the Carpathian Mountains on the border of Transylvania, Bukovina and Moldavia). The purpose of his mission is to provide legal support to Dracula for a real estate transaction overseen by Harker's employer, Peter Hawkins, of Exeter in England. At first enticed by Dracula's gracious manner, Harker soon discovers that he has become a prisoner in the castle. He also begins to see disquieting facets of Dracula's nocturnal life. One night while searching for a way out of the castle, and against Dracula's strict admonition not to venture outside his room at night, Harker falls under the spell of three wanton female vampires, \"the Sisters.\" He is saved at the last second by the Count, because he wants to keep Harker alive just long enough to obtain needed legal advice and teachings about England and London (Dracula's planned travel destination was to be among the \"teeming millions\"). Harker barely escapes from the castle with his life. Not long afterward, a Russian ship, the Demeter, having weighed anchor at Varna, runs aground on the shores of Whitby, England, during a fierce tempest. All of the crew are missing and presumed dead, and only one body is found, that of the captain tied to the ship's helm. The captain's log is recovered and tells of strange events that had taken place during the ship's journey. These events led to the gradual disappearance of the entire crew apparently owing to a malevolent presence on board the ill-fated ship. An animal described as a large dog is seen on the ship leaping ashore. The ship's cargo is described as silver sand and boxes of \"mould\", or earth, from Transylvania. Soon Dracula is tracking Harker's devoted fiancée, Wilhelmina \"Mina\" Murray, and her friend, Lucy Westenra. Lucy receives three marriage proposals in one day, from Dr. John Seward; Quincey Morris; and the Hon. Arthur Holmwood (later Lord Godalming). Lucy accepts Holmwood's proposal while turning down Seward and Morris, but all remain friends. There is a notable encounter between Dracula and Seward's patient Renfield, an insane man who means to consume insects, spiders, birds, and other creatures &mdash; in ascending order of size &mdash; in order to absorb their \"life force\". Renfield acts as a motion sensor, detecting Dracula's proximity and supplying clues accordingly. Lucy begins to waste away suspiciously. All of her suitors fret, and Seward calls in his old teacher, Professor Abraham Van Helsing from Amsterdam. Van Helsing immediately determines the cause of Lucy's condition but refuses to disclose it, knowing that Seward's faith in him will be shaken if he starts to speak of vampires. Van Helsing tries multiple blood transfusions, but they are clearly losing ground. On a night when Van Helsing must return to Amsterdam (and his message to Seward asking him to watch the Westenra household is delayed), Lucy and her mother are attacked by a wolf. Mrs. Westenra, who has a heart condition, dies of fright, and Lucy apparently dies soon after. Lucy is buried, but soon afterward the newspapers report children being stalked in the night by a \"bloofer lady\" (as they describe it), i.e. \"beautiful lady\". Van Helsing, knowing that this means Lucy has become a vampire, confides in Seward, Lord Godalming and Morris. The suitors and Van Helsing track her down, and after a disturbing confrontation between her vampiric self and Arthur, they stake her heart, behead her, and fill her mouth with garlic. Around the same time, Jonathan Harker arrives home from recuperation in Budapest (where Mina joined and married him after his escape from the castle); he and Mina also join the coalition, who turn their attentions to dealing with Dracula. After Dracula learns of Van Helsing's and the others' plot against him, he takes revenge by visiting – and feeding from – Mina at least three times. Dracula also feeds Mina his blood, creating a spiritual bond between them to control her. The only way to forestall this is to kill Dracula first. Mina slowly succumbs to the blood of the vampire that flows through her veins, switching back and forth from a state of consciousness to a state of semi-trance during which she is telepathically connected with Dracula. This telepathic connection is established to be two-way, in that the Count can influence Mina, but in doing so betrays to her awareness of his surroundings. After the group sterilizes all of his lairs in London by putting pieces of consecrated host in each box of earth, Dracula flees back to his castle in Transylvania, transported in a box with transfer and portage instructions forwarded, pursued by Van Helsing's group, who themselves are aided by Van Helsing hypnotizing Mina and questioning her about the Count. The group splits in three directions. Van Helsing goes to the Count's castle and kills his trio of brides, and shortly afterwards all converge on the Count just at sundown under the shadow of the castle. Harker and Quincey rush to Dracula's box, which is being transported by Gypsies. Harker shears Dracula through the throat with a Kukri while the mortally wounded Quincey, slashed by one of the crew, stabs the Count in the heart with a Bowie knife. Dracula crumbles to dust, and Mina is freed from his curse. The book closes with a note about Mina's and Jonathan's married life and the birth of their first-born son, whom they name after all four members of the party, but refer to only as Quincey in remembrance of their American friend."
    extractor = TripletExtractor()
    extractor.extract_from_full_text(text, 1)

if __name__ == "__main__":
    main()