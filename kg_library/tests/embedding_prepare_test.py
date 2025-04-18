import unittest
from kg_library.common import GraphData
from kg_library.models import EmbeddingPreprocessor


def create_test_graph() -> GraphData:
    graph = GraphData()

    graph.add_new_triplet("Leo Tolstoy", "is_a", "author", check_synonyms=True, head_feature="person", tail_feature="type")
    graph.add_new_triplet("Fyodor Dostoevsky", "is_a", "author", check_synonyms=True, head_feature="person", tail_feature="type")
    graph.add_new_triplet("Anton Chekhov", "is_a", "author", check_synonyms=True, head_feature="person", tail_feature="type")

    graph.add_new_triplet("War and Peace", "is_a", "novel", check_synonyms=True, head_feature="book", tail_feature="type")
    graph.add_new_triplet("Anna Karenina", "is_a", "novel", check_synonyms=True, head_feature="book", tail_feature="type")
    graph.add_new_triplet("Crime and Punishment", "is_a", "novel", check_synonyms=True, head_feature="book", tail_feature="type")
    graph.add_new_triplet("The Cherry Orchard", "is_a", "play", check_synonyms=True, head_feature="book", tail_feature="type")
    graph.add_new_triplet("The Brothers Karamazov", "is_a", "novel", check_synonyms=True, head_feature="book", tail_feature="type")
    graph.add_new_triplet("Idiot", "is_a", "novel", check_synonyms=True, head_feature="book", tail_feature="type")

    graph.add_new_triplet("Leo Tolstoy", "wrote", "War and Peace", check_synonyms=True, head_feature="person", tail_feature="book")
    graph.add_new_triplet("Leo Tolstoy", "wrote", "Anna Karenina", check_synonyms=True, head_feature="person", tail_feature="book")
    graph.add_new_triplet("Fyodor Dostoevsky", "wrote", "The Brothers Karamazov", check_synonyms=True, head_feature="person", tail_feature="book")
    graph.add_new_triplet("Fyodor Dostoevsky", "wrote", "Idiot", check_synonyms=True, head_feature="person", tail_feature="book")
    graph.add_new_triplet("Fyodor Dostoevsky", "wrote", "Crime and Punishment", check_synonyms=True, head_feature="person", tail_feature="book")
    graph.add_new_triplet("Anton Chekhov", "wrote", "The Cherry Orchard", check_synonyms=True, head_feature="person", tail_feature="book")

    graph.add_new_triplet("Pierre Bezukhov", "is_a", "character", check_synonyms=True, head_feature="character", tail_feature="type")
    graph.add_new_triplet("Anna Karenina", "is_a", "character", check_synonyms=True, head_feature="character", tail_feature="type")
    graph.add_new_triplet("Raskolnikov", "is_a", "character", check_synonyms=True, head_feature="character", tail_feature="type")
    graph.add_new_triplet("Ranevskaya", "is_a", "character", check_synonyms=True, head_feature="character", tail_feature="type")

    graph.add_new_triplet("Pierre Bezukhov", "appears_in", "War and Peace", check_synonyms=True, head_feature="character", tail_feature="book")
    graph.add_new_triplet("Anna Karenina", "appears_in", "Anna Karenina", check_synonyms=True, head_feature="character", tail_feature="book")
    graph.add_new_triplet("Raskolnikov", "appears_in", "Crime and Punishment", check_synonyms=True, head_feature="character", tail_feature="book")
    graph.add_new_triplet("Ranevskaya", "appears_in", "The Cherry Orchard", check_synonyms=True, head_feature="character", tail_feature="book")

    graph.add_new_triplet("Leo Tolstoy", "influenced", "Anton Chekhov", check_synonyms=True, head_feature="person", tail_feature="person")
    graph.add_new_triplet("Fyodor Dostoevsky", "influenced", "Leo Tolstoy", check_synonyms=True, head_feature="person", tail_feature="person")

    graph.add_new_triplet("War and Peace", "belongs_to_genre", "realism", check_synonyms=True, head_feature="book", tail_feature="genre")
    graph.add_new_triplet("Crime and Punishment", "belongs_to_genre", "psychological_fiction", check_synonyms=True, head_feature="book", tail_feature="genre")
    graph.add_new_triplet("The Cherry Orchard", "belongs_to_genre", "drama", check_synonyms=True, head_feature="book", tail_feature="genre")

    graph.add_loop_reversed_triplet()
    graph.print()

    return graph


class EmbeddingPrepareTest(unittest.TestCase):

    def test_feature_matrix(self):
        graph = create_test_graph()
        preprocessor = EmbeddingPreprocessor(graph)
        preprocessor.build_feature_matrix()
        print(preprocessor.feature_matrix)
        self.assertIsNotNone(preprocessor.feature_matrix)

    def test_creating_leaf_tensor(self):
        graph = create_test_graph()
        preprocessor = EmbeddingPreprocessor(graph)
        preprocessor.build_feature_matrix()
        preprocessor.build_hetero_graph()
        print(preprocessor.hetero_graph.to_dict())
        print(preprocessor.hetero_graph.metadata())
        self.assertIsNotNone(preprocessor.hetero_graph)

    def test_full_preprocessing(self):
        graph = create_test_graph()
        preprocessor = EmbeddingPreprocessor(graph)
        preprocessor.preprocess()
        print(f"feature matrix: {preprocessor.feature_matrix}")
        print(f"hetero data node types: {preprocessor.hetero_graph.node_types}")
        print(f"hetero data: {preprocessor.hetero_graph.to_dict()}")
        print(f"entity id: {preprocessor.entity_id}")
        print(f"relation id: {preprocessor.relation_id}")
        print(f"triplets: {preprocessor.split_triplets}")
        print(f"labels: {preprocessor.labels}")
        self.assertIsNotNone(preprocessor.split_triplets and preprocessor.labels)

if __name__ == '__main__':
    unittest.main()
