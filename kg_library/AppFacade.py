from typing import Dict, List
from kg_library.common import GraphData, WikidataExtractor
from kg_library.models import KnowledgeGraphExtractor, EmbeddingPreprocessor
from kg_library.common import data_frame
from kg_library import Neo4jConnection

class AppFacade:
    def __init__(self):
        self.knowledge_graph_extractor = KnowledgeGraphExtractor()
        self.neo4j_connection = Neo4jConnection()
        self.base_information_extractor = WikidataExtractor()
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
            if title != 'None' or author != 'None':
                self.graph.add_new_triplet(title, "author", author)
            if publication_date != 'None':
                self.graph.add_new_triplet(title, "published_in", publication_date)
            for genre in genres:
                if genre == 'None':
                    self.graph.add_new_triplet(title, "has_genre", genre)
            self.extract_plot_summary(i)
        #self.graph.add_loop_reversed_triplet()
        self.graph.print()
        self.graph.fill_database(self.neo4j_connection)


    def import_wikidata_information(self, info : Dict[str, List[Dict]], title):
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
            self.graph.add_new_triplet(title, "author", author["label"])
            self.graph.add_new_triplet(title, "author_id", author["id"])

        for publication_date in info["publication_dates"]:
            self.graph.add_new_triplet(title, "published_in", publication_date["label"])
            self.graph.add_new_triplet(title, "published_in_id", publication_date["id"])

        for country in info["countries"]:
            self.graph.add_new_triplet(title, "place", country["label"])
            self.graph.add_new_triplet(title, "place_id", country["id"])

        for language in info["languages"]:
            self.graph.add_new_triplet(title, "language", language["label"])
            self.graph.add_new_triplet(title, "language_id", language["id"])

        for character in info["characters"]:
            self.graph.add_new_triplet(title, "character", character["label"])
            self.graph.add_new_triplet(title, "character_id", character["id"])

        for location in info["locations"]:
            self.graph.add_new_triplet(title, "location", location["label"])
            self.graph.add_new_triplet(title, "location_id", location["id"])

    def extract_plot_summary(self, index):
        summary = self.dataset["Summarized Plot Summary"][index]
        if summary != 'None':
            extracted_triplets = self.knowledge_graph_extractor.extract_from_text(summary)
            self.knowledge_graph_extractor.print_knowledge_graph()
            for head, relation, tail in extracted_triplets:
                self.graph.add_new_triplet(head, relation, tail)

    def generate_graph(self):
        self.input_base_data()
