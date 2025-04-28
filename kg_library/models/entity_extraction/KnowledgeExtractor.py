from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from kg_library.utils import Preprocessing


class KnowledgeGraphExtractor:
    def __init__(self, model_name="Babelscape/mrebel-large", src_lang="en_XX", tgt_lang="tp_XX"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang=src_lang, tgt_lang=tgt_lang)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
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

    def __extract_triplets_typed(self, text: str):
        triplets = []
        current = 'x'
        subject = relation = object_ = object_type = subject_type = ''

        for token in text.replace("<s>", "").replace("<pad>", "").replace("</s>", "").replace("tp_XX", "").replace(
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

    def extract_from_text(self, text: str, num_beams=1, max_length=1024) -> list:
        model_inputs = self.tokenizer(text, max_length=max_length, padding=True, truncation=True, return_tensors="pt")
        print(text)
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
        all_triplets = []
        for sentence in decoded_preds:
            #print("Decoded sentence:", sentence)
            triplets = self.__extract_triplets_typed(sentence)
            all_triplets.extend(triplets)
            for t in triplets:
                self.triplets.add((t['head'], t['type'], t['tail'], t['head_type'], t['tail_type']))  # для GNN
        return all_triplets

    def extract_from_full_text(self, text: str) -> set:
        preprocessed_sentences = self.preprocessor.preprocessing(text)
        for sentence in preprocessed_sentences:
            self.extract_from_text(sentence)
        return self.triplets

    def print_knowledge_graph(self):
        print("Extracted Knowledge Graph:")
        for head, relation, tail, head_type, tail_type in sorted(self.triplets):
            print(f"  ({head} - [{head_type}]) -[{relation}]-> ({tail} - [{tail_type}])")


if __name__ == "__main__":
    text = "Old Major, the old boar on the Manor Farm, calls the animals on the farm for a meeting, where he compares the humans to parasites and teaches the animals a revolutionary song, 'Beasts of England'. When Major dies, two young pigs, Snowball and Napoleon, assume command and turn his dream into a philosophy. The animals revolt and drive the drunken and irresponsible Mr Jones from the farm, renaming it Animal Farm. They adopt Seven Commandments of Animal-ism, the most important of which is, All animals are equal. Snowball attempts to teach the animals reading and writing; food is plentiful, and the farm runs smoothly. The pigs elevate themselves to positions of leadership and set aside special food items, ostensibly for their personal health. Napoleon takes the pups from the farm dogs and trains them privately. Napoleon and Snowball struggle for leadership. When Snowball announces his plans to build a windmill, Napoleon has his dogs chase Snowball away and declares himself leader. Napoleon enacts changes to the governance structure of the farm, replacing meetings with a committee of pigs, who will run the farm. Using a young pig named Squealer as a mouthpiece, Napoleon claims credit for the windmill idea. The animals work harder with the promise of easier lives with the windmill. After a violent storm, the animals find the windmill annihilated. Napoleon and Squealer convince the animals that Snowball destroyed it, although the scorn of the neighbouring farmers suggests that its walls were too thin. Once Snowball becomes a scapegoat, Napoleon begins purging the farm with his dogs, killing animals he accuses of consorting with his old rival. He and the pigs abuse their power, imposing more control while reserving privileges for themselves and rewriting history, villainising Snowball and glorifying Napoleon. Squealer justifies every statement Napoleon makes, even the pigs' alteration of the Seven Commandments of Animalism to benefit themselves. 'Beasts of England' is replaced by an anthem glorifying Napoleon, who appears to be adopting the lifestyle of a man. The animals remain convinced that they are better off than they were when under Mr Jones. Squealer abuses the animals' poor memories and invents numbers to show their improvement. Mr Frederick, one of the neighbouring farmers, attacks the farm, using blasting powder to blow up the restored windmill. Though the animals win the battle, they do so at great cost, as many, including Boxer the workhorse, are wounded. Despite his injuries, Boxer continues working harder and harder, until he collapses while working on the windmill. Napoleon sends for a van to take Boxer to the veterinary surgeon's, explaining that better care can be given there. Benjamin, the cynical donkey, who could read as well as any pig, notices that the van belongs to a knacker, and attempts to mount a rescue; but the animals' attempts are futile. Squealer reports that the van was purchased by the hospital and the writing from the previous owner had not been repainted. He recounts a tale of Boxer's death in the hands of the best medical care. Years pass, and the pigs learn to walk upright, carry whips and wear clothes. The Seven Commandments are reduced to a single phrase: All animals are equal, but some animals are more equal than others. Napoleon holds a dinner party for the pigs and the humans of the area, who congratulate Napoleon on having the hardest-working but least fed animals in the country. Napoleon announces an alliance with the humans, against the labouring classes of both worlds. He abolishes practices and traditions related to the Revolution, and changes the name of the farm to The Manor Farm. The animals, overhearing the conversation, notice that the faces of the pigs have begun changing. During a poker match, an argument breaks out between Napoleon and Mr Pilkington, and the animals realise that the faces of the pigs look like the faces of humans, and no one can tell the difference between them. The pigs Snowball, Napoleon, and Squealer adapt Old Major's ideas into an actual philosophy, which they formally name Animalism. Soon after, Napoleon and Squealer indulge in the vices of humans (drinking alcohol, sleeping in beds, trading). Squealer is employed to alter the Seven Commandments to account for this humanisation, an allusion to the Soviet government's revising of history in order to exercise control of the people's beliefs about themselves and their society. The original commandments are: # Whatever goes upon two legs is an enemy. # Whatever goes upon four legs, or has wings, is a friend. # No animal shall wear clothes. # No animal shall sleep in a bed. # No animal shall drink alcohol. # No animal shall kill any other animal. # All animals are equal. Later, Napoleon and his pigs secretly revise some commandments to clear them of accusations of law-breaking (such as No animal shall drink alcohol having to excess appended to it and No animal shall sleep in a bed with with sheets added to it). The changed commandments are as follows, with the changes bolded: * 4 No animal shall sleep in a bed with sheets. * 5 No animal shall drink alcohol to excess. * 6 No animal shall kill any other animal without cause. Eventually these are replaced with the maxims, All animals are equal, but some animals are more equal than others, and Four legs good, two legs better! as the pigs become more human. This is an ironic twist to the original purpose of the Seven Commandments, which were supposed to keep order within Animal Farm by uniting the animals together against the humans, and prevent animals from following the humans' evil habits. Through the revision of the commandments, Orwell demonstrates how simply political dogma can be turned into malleable propaganda."
    text2 = " Old Major, an old boar, inspires a farm animal rebellion against humans, leading to the creation of Animal Farm. After his death, pigs Snowball and Napoleon lead, but Napoleon eventually seizes power, exiling Snowball. The pigs exploit the animals, altering commandments to suit their desires, and become indistinguishable from humans. The farm's ideals decay into tyranny, with the maxim All animals are equal, but some animals are more equal than others symbolizing the betrayal of the revolution."
    extractor = KnowledgeGraphExtractor()
    special_tokens = extractor.tokenizer.additional_special_tokens

    print(special_tokens)
    extractor.extract_from_full_text(text2)
    extractor.print_knowledge_graph()
