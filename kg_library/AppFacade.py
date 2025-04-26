from kg_library.common import GraphData, WikidataExtractor
from kg_library.models import KnowledgeGraphExtractor, GraphTrainer, GraphNN, EmbeddingPreprocessor, create_dataloader
from kg_library.common import data_frame
from kg_library import Neo4jConnection


class AppFacade:
    def __init__(self):
        self.knowledge_graph_extractor = KnowledgeGraphExtractor()
        self.neo4j_connection = Neo4jConnection()
        self.base_information_extractor = WikidataExtractor()
        self.graph_trainer : GraphTrainer
        self.preprocessor : EmbeddingPreprocessor
        self.model : GraphNN
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
        extracted_triplets = set()
        if plot != 'None':
            extracted_triplets.add(self.knowledge_graph_extractor.extract_from_full_text(plot))
        return extracted_triplets

    def generate_graph_for_learning(self):
        self.input_base_data()

    def learning_process(self):
        self.generate_graph_for_learning()
        self.preprocessor = EmbeddingPreprocessor(self.graph)
        self.preprocessor.preprocess()
        self.model = GraphNN(self.preprocessor)
        train_loader, test_loader, val_loader = create_dataloader(self.preprocessor, batch_size=128)
        self.graph_trainer = GraphTrainer(self.model, train_loader, val_loader, epochs=10, lr=0.0005)
        self.graph_trainer.train()
        val_auc = self.graph_trainer.evaluate(self.graph_trainer.val_loader)
        print(f"Final Val AUC: {val_auc}")
        self.graph_trainer.save_with_config()

    def generate_graph(self, text : str):
        pass
