from typing import Optional
from kg_library.common import GraphData, WikidataExtractor, data_frame, GraphJSON
from kg_library.models import KnowledgeGraphExtractor, GraphTrainer, GraphNN, EmbeddingPreprocessor, create_dataloader
from kg_library import Neo4jConnection
from kg_library.models.evaluation.TripletEvaluator import TripletEvaluator
from kg_library.utils import AudioProcessor
import torch

class AppFacade:
    def __init__(self):
        self.knowledge_graph_extractor = KnowledgeGraphExtractor()
        #self.neo4j_connection = Neo4jConnection()
        self.base_information_extractor = WikidataExtractor()
        self.graph_trainer : Optional[GraphTrainer] = None
        self.preprocessor : Optional[EmbeddingPreprocessor] = None
        self.model : Optional[GraphNN] = None
        self.audio_processor = AudioProcessor()
        self.graph = GraphData()
        self.dataset = data_frame
        self.size = 200

    def input_base_data(self):
        for i in range(self.size):
            title = self.dataset['Book title'][i]
            self.import_wikidata_information(self.base_information_extractor.get_book_info(title), title)
            title = self.dataset['Book title'][i]
            print(title)
            author = self.dataset['Author'][i]
            print(author)
            genres = self.dataset['Parsed Genres'][i].split(', ')
            print(genres)
            publication_date = self.dataset['Publication date'][i]
            print(publication_date)
            if title is not None or author is not None:
                self.graph.add_new_triplet(title, "author", author, check_synonyms=False, head_feature="title",
                                           tail_feature="person")
            if publication_date is not None:
                self.graph.add_new_triplet(title, "published_in", publication_date, check_synonyms=False,
                                           head_feature="title", tail_feature="date")
            for genre in genres:
                self.graph.add_new_triplet(title, "has_genre", genre, check_synonyms=False, head_feature="title",
                                               tail_feature="genre")
            self.extract_plot_summary(i)
        self.graph.add_loop_reversed_triplet()
        self.graph.print()
        #self.graph.fill_database(self.neo4j_connection)

    def import_wikidata_information(self, info: dict[str, list[dict]], title):
        print(info)
        '''
        book_info = {
            "authors": [],
            "publication_dates": [],
            "countries": [],
            "languages": [],
            "characters": [],
            "locations": []
        }
        '''
        for author in info["authors"]:
            self.graph.add_new_triplet(title, "author", author["label"], check_synonyms=True, head_feature="title",
                                       tail_feature="person")

        for publication_date in info["publication_dates"]:
            self.graph.add_new_triplet(title, "published_in", publication_date["label"], check_synonyms=True,
                                       head_feature="title", tail_feature="date")

        for country in info["countries"]:
            self.graph.add_new_triplet(title, "place", country["label"], check_synonyms=False, head_feature="title",
                                       tail_feature="country")

        for language in info["languages"]:
            self.graph.add_new_triplet(title, "language", language["label"], check_synonyms=False, head_feature="title",
                                       tail_feature="language")

        for character in info["characters"]:
            self.graph.add_new_triplet(title, "character", character["label"], check_synonyms=True,
                                       head_feature="title", tail_feature="character")

        for location in info["locations"]:
            self.graph.add_new_triplet(title, "location", location["label"], check_synonyms=False, head_feature="title",
                                       tail_feature="location")

    def extract_plot_summary(self, index):
        summary = self.dataset["Summarized Plot Summary"][index]
        if summary != 'None':
            extracted_triplets = self.knowledge_graph_extractor.extract_from_full_text(summary)
            for head, relation, tail, head_feature, tail_feature in extracted_triplets:
                if head is not None or relation is not None or tail is not None:
                    if head_feature is None:
                        head_feature = "default"
                    if tail_feature is None:
                        tail_feature = "default"
                    if head_feature.lower() == "person":
                        head_feature = "character"
                    if tail_feature.lower() == "person":
                        tail_feature = "character"
                    self.graph.add_new_triplet(head, relation, tail, check_synonyms=True, head_feature=head_feature.lower(), tail_feature=tail_feature.lower())

    def extract_plot(self, index) -> set:
        plot = self.dataset["Plot summary"][index]
        if plot != 'None':
            return self.knowledge_graph_extractor.extract_from_full_text(plot)
        else:
            return set()

    def generate_graph_for_learning(self, load_model_from_file=False, load_triplets_from_file=False):
        if load_model_from_file:
            self.load_model_for_finetune()
        else:
            self.input_base_data()
            self.learning_process()
        #self.find_internal_links()
        extra_triplets = set()
        if not load_triplets_from_file:
            for i in range(self.size):
                print(i, end="\n\n")
                extra_triplets.update(self.extract_plot(i))
            self.knowledge_graph_extractor.save_triplets_to_json()
        else:
            self.knowledge_graph_extractor.load_triplets_from_json()
            extra_triplets = self.knowledge_graph_extractor.triplets
        self.finetune_model(extra_triplets)

    def learning_process(self):
        self.generate_graph_for_learning()
        self.preprocessor = EmbeddingPreprocessor(self.graph)
        self.preprocessor.preprocess(self.knowledge_graph_extractor.type_map.values())

        self.model = GraphNN(self.preprocessor)
        train_loader, test_loader, val_loader = create_dataloader(self.preprocessor, batch_size=128)
        self.graph_trainer = GraphTrainer(self.model, train_loader, val_loader, epochs=10, lr=0.0005)
        self.graph_trainer.train()
        val_auc = self.graph_trainer.evaluate(self.graph_trainer.val_loader)
        print(f"Final Val AUC: {val_auc}")
        self.graph_trainer.save_with_config()

    def generate_graph_from_audio(self, audio : str):
        text = self.audio_processor.transform_to_text(audio)
        self.generate_graph_from_text(text)

    def generate_graph_from_text(self, text : str, confidence_threshold = 0.65, find_internal_links = True) -> GraphData:
        raw_triplets = self.knowledge_graph_extractor.extract_from_full_text(text)
        temp_graph = GraphData()
        if self.model is None:
            try:
                print("Загрузка модели для фильтрации триплетов...")
                self.load_model(model_path="model_finetune.pt", map_location='cuda')
                print("Модель успешно загружена")
            except Exception as e:
                print(f"Невозможно загрузить модель: {str(e)}")
                for head, relation, tail, head_feature, tail_feature in raw_triplets:
                    temp_graph.add_new_triplet(
                        head=head,
                        relation=relation,
                        tail=tail,
                        check_synonyms=True,
                        head_feature=head_feature.lower() if head_feature else "default",
                        tail_feature=tail_feature.lower() if tail_feature else "default"
                    )

                self.graph = temp_graph
                return temp_graph
        evaluator = TripletEvaluator(self.model)
        filtered_triplets = []
        print(f"Фильтрация {len(raw_triplets)} триплетов")
        for head, relation, tail, head_feature, tail_feature in raw_triplets:
            head_id = self.preprocessor.entity_id.get(head, None)
            tail_id = self.preprocessor.entity_id.get(tail, None)
            score = evaluator.score_new_triplet(
                head_feature=head_feature,
                tail_feature=tail_feature,
                head_name=head,
                tail_name=tail,
                relation=relation,
                head_id=head_id,
                tail_id=tail_id,
                graph=self.preprocessor.hetero_graph
            )
            probability = torch.sigmoid(torch.tensor(score)).item()

            if probability > confidence_threshold:
                print(f"Принят триплет: {head} - [{relation}] -> {tail} (уверенность: {probability:.4f})")
                filtered_triplets.append((head, relation, tail, head_feature, tail_feature))

                temp_graph.add_new_triplet(
                    head=head,
                    relation=relation,
                    tail=tail,
                    check_synonyms=True,
                    head_feature=head_feature.lower() if head_feature else "default",
                    tail_feature=tail_feature.lower() if tail_feature else "default"
                )
            else:
                print(f"Отклонен триплет: {head} - [{relation}] -> {tail} (уверенность: {probability:.4f})")

        if find_internal_links and filtered_triplets:
            print("Поиск потенциальных внутренних связей между сущностями...")

            temp_graph.add_loop_reversed_triplet()

            temp_preprocessor = EmbeddingPreprocessor(temp_graph)
            temp_preprocessor.preprocess(self.knowledge_graph_extractor.type_map.values())

            temp_model = GraphNN.load_model(
                model_path=self.model_path if hasattr(self, 'model_path') else "model_with_config.pt",
                map_location=self.model.device,
                preprocessor=temp_preprocessor
            )

            temp_evaluator = TripletEvaluator(temp_model)
            potential_links = temp_evaluator.link_prediction_in_graph(
                threshold=confidence_threshold,
                top_k=8
            )
            for link in potential_links:
                head, relation, tail = link['head'], link['relation'], link['tail']
                probability = link['score']

                if not temp_graph.has_triplet(head, relation, tail):
                    head_node = temp_graph.find_node(head)
                    tail_node = temp_graph.find_node(tail)

                    if head_node and tail_node:
                        print(
                            f"Найдена внутренняя связь: {head} - [{relation}] -> {tail} (уверенность: {probability:.4f})")
                        temp_graph.add_new_triplet(
                            head=head,
                            relation=relation,
                            tail=tail,
                            check_synonyms=False,
                            head_feature=head_node.feature,
                            tail_feature=tail_node.feature
                        )

            self.graph = temp_graph

            print(f"Итоговый граф содержит {len(self.graph.triplets)} триплетов")
            self.graph.print()

        return self.graph

    def find_internal_links(self):
        evaluator = TripletEvaluator(self.model)
        print("Internal links:")
        result = evaluator.link_prediction_in_graph()
        print(result, end='\n\n')

    def load_model(self, model_path="model.pt", map_location='cuda'):
        checkpoint = torch.load(model_path, map_location=map_location)
        if "graph" in checkpoint:
            self.graph = GraphJSON.load(checkpoint["graph"])
            self.preprocessor = EmbeddingPreprocessor(self.graph)
            self.preprocessor.load_config(checkpoint["preprocessor_config"])
        self.model = GraphNN.load_model(model_path, map_location=map_location, preprocessor=self.preprocessor)

    def load_model_for_finetune(self, model_path="model_with_config.pt", map_location='cuda'):
        checkpoint = torch.load(model_path, map_location=map_location, weights_only=False)
        self.graph = GraphJSON.load(checkpoint["graph"])
        self.preprocessor = EmbeddingPreprocessor(self.graph)
        self.preprocessor.load_config(checkpoint["preprocessor_config"])
        train_loader, test_loader, val_loader = create_dataloader(self.preprocessor, batch_size=64)
        self.graph_trainer = GraphTrainer.load_model_for_training(self.preprocessor, model_path, map_location=map_location, train_loader=train_loader, val_loader=val_loader)
        self.model = self.graph_trainer.model

    def finetune_model(self, new_triplets, confidence_threshold=0.65) -> dict:
        if self.graph_trainer is None:
            self.load_model_for_finetune()
        updated_graph = self.graph.clone()
        evaluator = TripletEvaluator(self.model)
        added_triplets = 0
        print("New triplets:", new_triplets)
        for head, relation, tail, head_feature, tail_feature in new_triplets:
            if updated_graph.has_triplet(head, relation, tail):
                print(f"Already has triplet {head} - [{relation}] -> {tail}")
                continue
            head_id = self.preprocessor.entity_id.get(head, None)
            tail_id = self.preprocessor.entity_id.get(tail, None)

            score = evaluator.score_new_triplet(
                head_feature=head_feature,
                tail_feature=tail_feature,
                head_name=head,
                tail_name=tail,
                relation=relation,
                head_id=head_id,
                tail_id=tail_id,
                graph=self.preprocessor.hetero_graph
            )

            probability = torch.sigmoid(torch.tensor(score)).item()

            if probability > confidence_threshold:
                print(f"Adding new triplet {head} - [{relation}] -> {tail} (score: {probability:.4f})")
                updated_graph.add_new_triplet(
                    head=head,
                    relation=relation,
                    tail=tail,
                    check_synonyms=True,
                    head_feature=head_feature,
                    tail_feature=tail_feature
                )
                added_triplets += 1
            else:
                print(f"Discarding new triplet {head} - [{relation}] -> {tail} (score: {probability:.4f})")

        if added_triplets == 0:
            return {}

        updated_preprocessor = EmbeddingPreprocessor(updated_graph)
        updated_preprocessor.preprocess(self.knowledge_graph_extractor.type_map.values())
        updated_train_loader, updated_test_loader, updated_val_loader = create_dataloader(updated_preprocessor, batch_size=64)

        new_model = GraphNN(
            preprocessor=updated_preprocessor,
            hidden_dim=self.model.get_config()["hidden_dim"],
            num_layers=self.model.get_config()["num_layers"],
            dropout=self.model.get_config()["dropout"]
        )

        self._transfer_weights(self.model, new_model, updated_preprocessor)

        trainer = GraphTrainer(new_model, updated_train_loader, updated_val_loader, epochs=15)

        print(f"Начинаем дообучение на {added_triplets} новых триплетах...")
        best_auc = trainer.train()

        test_auc = trainer.evaluate(updated_val_loader)
        print(f"Финальный Test AUC после дообучения: {test_auc:.4f}")

        trainer.save_with_config("model_finetune.pt", "graph_finetune.json")

        self.model = new_model
        self.graph = updated_graph
        self.preprocessor = updated_preprocessor

        return {
            "added_triplets": added_triplets,
            "final_val_auc": best_auc,
            "final_test_auc": test_auc
        }

    def _transfer_weights(self, model, new_model, updated_preprocessor):
        with torch.no_grad():
            for entity, old_id in model.preprocessor.entity_id.items():
                if entity in updated_preprocessor.entity_id:
                    new_id = updated_preprocessor.entity_id[entity]
                    new_model.entity_embedding.weight[new_id] = model.entity_embedding.weight[old_id]

            for relation, old_id in model.preprocessor.relation_id.items():
                if relation in updated_preprocessor.relation_id:
                    new_id = updated_preprocessor.relation_id[relation]
                    new_model.relation_embedding.weight[new_id] = model.relation_embedding.weight[old_id]

            for name, param in model.named_parameters():
                if "embedding" not in name:
                    try:
                        new_param = new_model.get_parameter(name)
                        if param.shape == new_param.shape:
                            new_param.copy_(param)
                    except Exception:
                        continue