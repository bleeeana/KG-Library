import unittest
from kg_library.common import GraphData
from kg_library.models import EmbeddingPreprocessor


def create_test_graph() -> GraphData:
    graph = GraphData()

    graph.add_new_triplet("Leo Tolstoy", "is_a", "author")
    graph.add_new_triplet("Fyodor Dostoevsky", "is_a", "author")
    graph.add_new_triplet("Anton Chekhov", "is_a", "author")

    graph.add_new_triplet("War and Peace", "is_a", "novel")
    graph.add_new_triplet("Anna Karenina", "is_a", "novel")
    graph.add_new_triplet("Crime and Punishment", "is_a", "novel")
    graph.add_new_triplet("The Cherry Orchard", "is_a", "play")

    graph.add_new_triplet("Leo Tolstoy", "wrote", "War and Peace")
    graph.add_new_triplet("Leo Tolstoy", "wrote", "Anna Karenina")
    graph.add_new_triplet("Fyodor Dostoevsky", "wrote", "Crime and Punishment")
    graph.add_new_triplet("Anton Chekhov", "wrote", "The Cherry Orchard")

    graph.add_new_triplet("Pierre Bezukhov", "is_a", "character")
    graph.add_new_triplet("Anna Karenina", "is_a", "character")
    graph.add_new_triplet("Raskolnikov", "is_a", "character")
    graph.add_new_triplet("Ranevskaya", "is_a", "character")

    graph.add_new_triplet("Pierre Bezukhov", "appears_in", "War and Peace")
    graph.add_new_triplet("Anna Karenina", "appears_in", "Anna Karenina")
    graph.add_new_triplet("Raskolnikov", "appears_in", "Crime and Punishment")
    graph.add_new_triplet("Ranevskaya", "appears_in", "The Cherry Orchard")

    graph.add_new_triplet("Leo Tolstoy", "influenced", "Anton Chekhov")
    graph.add_new_triplet("Fyodor Dostoevsky", "influenced", "Leo Tolstoy")

    graph.add_new_triplet("War and Peace", "belongs_to_genre", "realism")
    graph.add_new_triplet("Crime and Punishment", "belongs_to_genre", "psychological_fiction")
    graph.add_new_triplet("The Cherry Orchard", "belongs_to_genre", "drama")

    graph.add_loop_reversed_triplet()
    graph.print()

    return graph

class EmbeddingPrepareTest(unittest.TestCase):

    def test_feature_matrix(self):
        graph = create_test_graph()
        preprocessor = EmbeddingPreprocessor(graph)
        _, _, tensor = preprocessor.build_feature_matrix()
        print(tensor)
        self.assertIsNotNone(tensor)

    def test_creating_leaf_tensor(self):
        graph = create_test_graph()
        preprocessor = EmbeddingPreprocessor(graph)
        hetero_graph, _, _, _ = preprocessor.build_hetero_graph()
        print(hetero_graph.to_dict())
        print(hetero_graph.metadata())
        self.assertIsNotNone(hetero_graph)

    def test_build_training_data(self):
        graph = create_test_graph()
        preprocessor = EmbeddingPreprocessor(graph)
        triplets, labels, entity_id, _ = preprocessor.prepare_training_data()
        print(f"triplets: {triplets}\n\n labels: {labels}\n\n entity_id: {entity_id}")
        self.assertIsNotNone(graph)

if __name__ == '__main__':
    unittest.main()
