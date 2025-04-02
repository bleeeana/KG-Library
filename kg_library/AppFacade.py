from kg_library.common import GraphData
from kg_library.models.entity_extraction.KnowledgeExtractor import KnowledgeGraphExtractor
from kg_library.common import data_frame
from kg_library import Neo4jConnection

class AppFacade:
    def __init__(self):
        self.knowledge_graph_extractor = KnowledgeGraphExtractor()
        self.neo4j_connection = Neo4jConnection()
        self.graph = GraphData()
        self.dataset = data_frame
        self.size = 2

    def input_base_data(self):
        for i in range(self.size):
            title = self.dataset['Book title'][i]
            print(title)
            author = self.dataset['Author'][i]
            print(author)
            genres = self.dataset['Parsed Genres'][i].split(', ')
            print(genres)
            publication_date = self.dataset['Publication date'][i]
            print(publication_date)
            if title != 'None' or author != 'None':
                self.graph.add_new_triplet(title, "write:reversed", author)
                self.graph.add_new_triplet(author, "write", title)
            if publication_date != 'None':
                self.graph.add_new_triplet(title, "published_in", publication_date)
                self.graph.add_new_triplet(publication_date, "published_in:reversed", title)
            for genre in genres:
                if genre == 'None':
                    self.graph.add_new_triplet(title, "has_genre", genre)
                    self.graph.add_new_triplet(genre, "has_genre:reversed", title)
            self.extract_plot_summary(i)
        self.graph.fill_database(self.neo4j_connection)

    def extract_plot_summary(self, index):
        summary = self.dataset["Summarized Plot Summary"][index]
        if summary != 'None':
            extracted_triplets = self.knowledge_graph_extractor.extract_from_text(summary)
            self.knowledge_graph_extractor.print_knowledge_graph()
            for head, relation, tail in extracted_triplets:
                self.graph.add_new_triplet(head, relation, tail)
                self.graph.add_new_triplet(tail, f"{relation}:reversed", head)


    def generate_graph(self):
        self.input_base_data()