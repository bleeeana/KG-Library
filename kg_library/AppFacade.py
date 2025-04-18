from kg_library.common import GraphData, WikidataExtractor
from kg_library.models import KnowledgeGraphExtractor
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
                self.graph.add_new_triplet(title, "author", author, check_synonyms=True, head_feature="title", tail_feature="author")
            if publication_date != 'None':
                self.graph.add_new_triplet(title, "published_in", publication_date, check_synonyms=True, head_feature="title", tail_feature="date")
            for genre in genres:
                if genre == 'None':
                    self.graph.add_new_triplet(title, "has_genre", genre, check_synonyms=True, head_feature="title", tail_feature="genre")
            self.extract_plot_summary(i)
        #self.graph.add_loop_reversed_triplet()
        self.graph.print()
        self.graph.fill_database(self.neo4j_connection)


    def import_wikidata_information(self, info : dict[str, list[dict]], title):
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
            self.graph.add_new_triplet(title, "author", author["label"], check_synonyms=True, head_feature="title", tail_feature="person")
            self.graph.add_new_triplet(title, "author_id", author["id"], check_synonyms=True, head_feature="title", tail_feature="id")

        for publication_date in info["publication_dates"]:
            self.graph.add_new_triplet(title, "published_in", publication_date["label"], check_synonyms=True, head_feature="title", tail_feature="date")
            self.graph.add_new_triplet(title, "published_in_id", publication_date["id"], check_synonyms=True, head_feature="title", tail_feature="id")

        for country in info["countries"]:
            self.graph.add_new_triplet(title, "place", country["label"], check_synonyms=True, head_feature="title", tail_feature="country")
            self.graph.add_new_triplet(title, "place_id", country["id"], check_synonyms=True, head_feature="title", tail_feature="id")

        for language in info["languages"]:
            self.graph.add_new_triplet(title, "language", language["label"], check_synonyms=True, head_feature="title", tail_feature="language")
            self.graph.add_new_triplet(title, "language_id", language["id"], check_synonyms=True, head_feature="title", tail_feature="id")

        for character in info["characters"]:
            self.graph.add_new_triplet(title, "character", character["label"], check_synonyms=True, head_feature="title", tail_feature="character")
            self.graph.add_new_triplet(title, "character_id", character["id"], check_synonyms=True, head_feature="title", tail_feature="id")

        for location in info["locations"]:
            self.graph.add_new_triplet(title, "location", location["label"], check_synonyms=True, head_feature="title", tail_feature="location")
            self.graph.add_new_triplet(title, "location_id", location["id"], check_synonyms=True, head_feature="title", tail_feature="id")

    def extract_plot_summary(self, index):
        summary = self.dataset["Summarized Plot Summary"][index]
        if summary != 'None':
            extracted_triplets = self.knowledge_graph_extractor.extract_from_text(summary)
            self.knowledge_graph_extractor.print_knowledge_graph()
            for head, relation, tail in extracted_triplets:
                self.graph.add_new_triplet(head, relation, tail)

    def generate_graph(self):
        self.input_base_data()
