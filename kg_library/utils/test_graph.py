from kg_library.common import GraphData


def create_mini_test_graph() -> GraphData:
    graph = GraphData()

    graph.add_new_triplet("Leo Tolstoy", "is_a", "author", check_synonyms=False, head_feature="person",
                          tail_feature="type")
    graph.add_new_triplet("Fyodor Dostoevsky", "is_a", "author", check_synonyms=False, head_feature="person",
                          tail_feature="type")
    graph.add_new_triplet("Anton Chekhov", "is_a", "author", check_synonyms=False, head_feature="person",
                          tail_feature="type")
    graph.add_new_triplet("Alexander Pushkin", "is_a", "author", check_synonyms=False, head_feature="person",
                          tail_feature="type")

    graph.add_new_triplet("War and Peace", "is_a", "novel", check_synonyms=False, head_feature="book",
                          tail_feature="type")
    graph.add_new_triplet("Anna Karenina", "is_a", "novel", check_synonyms=False, head_feature="book",
                          tail_feature="type")
    graph.add_new_triplet("Crime and Punishment", "is_a", "novel", check_synonyms=False, head_feature="book",
                          tail_feature="type")
    graph.add_new_triplet("The Cherry Orchard", "is_a", "play", check_synonyms=False, head_feature="book",
                          tail_feature="type")
    graph.add_new_triplet("Eugene Onegin", "is_a", "poem", check_synonyms=False, head_feature="book",
                          tail_feature="type")

    graph.add_new_triplet("Leo Tolstoy", "wrote", "War and Peace", check_synonyms=False, head_feature="person",
                          tail_feature="book")
    graph.add_new_triplet("Leo Tolstoy", "wrote", "Anna Karenina", check_synonyms=False, head_feature="person",
                          tail_feature="book")
    graph.add_new_triplet("Fyodor Dostoevsky", "wrote", "Crime and Punishment", check_synonyms=False,
                          head_feature="person", tail_feature="book")
    graph.add_new_triplet("Anton Chekhov", "wrote", "The Cherry Orchard", check_synonyms=False, head_feature="person",
                          tail_feature="book")
    graph.add_new_triplet("Alexander Pushkin", "wrote", "Eugene Onegin", check_synonyms=False, head_feature="person",
                          tail_feature="book")

    graph.add_new_triplet("Pierre Bezukhov", "is_a", "character", check_synonyms=False, head_feature="character",
                          tail_feature="type")
    graph.add_new_triplet("Anna Karenina", "is_a", "character", check_synonyms=False, head_feature="character",
                          tail_feature="type")
    graph.add_new_triplet("Raskolnikov", "is_a", "character", check_synonyms=False, head_feature="character",
                          tail_feature="type")
    graph.add_new_triplet("Ranevskaya", "is_a", "character", check_synonyms=False, head_feature="character",
                          tail_feature="type")
    graph.add_new_triplet("Eugene Onegin", "is_a", "character", check_synonyms=False, head_feature="character",
                          tail_feature="type")
    graph.add_new_triplet("Tatyana Larina", "is_a", "character", check_synonyms=False, head_feature="character",
                          tail_feature="type")

    graph.add_new_triplet("Pierre Bezukhov", "appears_in", "War and Peace", check_synonyms=False,
                          head_feature="character", tail_feature="book")
    graph.add_new_triplet("Anna Karenina", "appears_in", "Anna Karenina", check_synonyms=False,
                          head_feature="character", tail_feature="book")
    graph.add_new_triplet("Raskolnikov", "appears_in", "Crime and Punishment", check_synonyms=False,
                          head_feature="character", tail_feature="book")
    graph.add_new_triplet("Ranevskaya", "appears_in", "The Cherry Orchard", check_synonyms=False,
                          head_feature="character", tail_feature="book")
    graph.add_new_triplet("Eugene Onegin", "appears_in", "Eugene Onegin", check_synonyms=False,
                          head_feature="character", tail_feature="book")
    graph.add_new_triplet("Tatyana Larina", "appears_in", "Eugene Onegin", check_synonyms=False,
                          head_feature="character", tail_feature="book")

    graph.add_new_triplet("Leo Tolstoy", "influenced", "Anton Chekhov", check_synonyms=False, head_feature="person",
                          tail_feature="person")
    graph.add_new_triplet("Fyodor Dostoevsky", "influenced", "Leo Tolstoy", check_synonyms=False, head_feature="person",
                          tail_feature="person")
    graph.add_new_triplet("Alexander Pushkin", "influenced", "Leo Tolstoy", check_synonyms=False, head_feature="person",
                          tail_feature="person")

    graph.add_new_triplet("War and Peace", "belongs_to_genre", "realism", check_synonyms=False, head_feature="book",
                          tail_feature="genre")
    graph.add_new_triplet("Anna Karenina", "belongs_to_genre", "realism", check_synonyms=False, head_feature="book",
                          tail_feature="genre")
    graph.add_new_triplet("Crime and Punishment", "belongs_to_genre", "psychological_fiction", check_synonyms=False,
                          head_feature="book", tail_feature="genre")
    graph.add_new_triplet("The Cherry Orchard", "belongs_to_genre", "drama", check_synonyms=False, head_feature="book",
                          tail_feature="genre")
    graph.add_new_triplet("Eugene Onegin", "belongs_to_genre", "poetry", check_synonyms=False, head_feature="book",
                          tail_feature="genre")

    graph.add_new_triplet("Leo Tolstoy", "lived_during", "19th_century", check_synonyms=False, head_feature="person",
                          tail_feature="time_period")
    graph.add_new_triplet("Fyodor Dostoevsky", "lived_during", "19th_century", check_synonyms=False,
                          head_feature="person", tail_feature="time_period")
    graph.add_new_triplet("Anton Chekhov", "lived_during", "19th_century", check_synonyms=False, head_feature="person",
                          tail_feature="time_period")
    graph.add_new_triplet("Alexander Pushkin", "lived_during", "19th_century", check_synonyms=False,
                          head_feature="person", tail_feature="time_period")

    graph.add_new_triplet("Leo Tolstoy", "has_nationality", "Russian", check_synonyms=False, head_feature="person",
                          tail_feature="nationality")
    graph.add_new_triplet("Fyodor Dostoevsky", "has_nationality", "Russian", check_synonyms=False,
                          head_feature="person", tail_feature="nationality")
    graph.add_new_triplet("Anton Chekhov", "has_nationality", "Russian", check_synonyms=False, head_feature="person",
                          tail_feature="nationality")
    graph.add_new_triplet("Alexander Pushkin", "has_nationality", "Russian", check_synonyms=False,
                          head_feature="person", tail_feature="nationality")

    graph.add_new_triplet("War and Peace", "published_in", "1869", check_synonyms=False, head_feature="book",
                          tail_feature="year")
    graph.add_new_triplet("Anna Karenina", "published_in", "1877", check_synonyms=False, head_feature="book",
                          tail_feature="year")
    graph.add_new_triplet("Crime and Punishment", "published_in", "1866", check_synonyms=False, head_feature="book",
                          tail_feature="year")
    graph.add_new_triplet("The Cherry Orchard", "published_in", "1904", check_synonyms=False, head_feature="book",
                          tail_feature="year")
    graph.add_new_triplet("Eugene Onegin", "published_in", "1833", check_synonyms=False, head_feature="book",
                          tail_feature="year")

    graph.add_new_triplet("Pierre Bezukhov", "friend_of", "Natasha Rostova", check_synonyms=False,
                          head_feature="character", tail_feature="character")
    graph.add_new_triplet("Eugene Onegin", "rejected", "Tatyana Larina", check_synonyms=False, head_feature="character",
                          tail_feature="character")

    graph.add_new_triplet("Nina Berberova", "is_a", "author", check_synonyms=False, head_feature="person", tail_feature="type")
    graph.add_new_triplet("Nina Berberova", "lived_during", "20th_century", check_synonyms=False, head_feature="person", tail_feature="time_period")
    graph.add_new_triplet("Nina Berberova", "wrote", "The Accompanist", check_synonyms=False, head_feature="person", tail_feature="book")
    graph.add_new_triplet("The Accompanist", "is_a", "novel", check_synonyms=False, head_feature="book", tail_feature="type")
    graph.add_new_triplet("The Accompanist", "belongs_to_genre", "realism", check_synonyms=False, head_feature="book", tail_feature="genre")
    graph.add_new_triplet("The Accompanist", "set_in", "Saint_Petersburg", check_synonyms=False, head_feature="book", tail_feature="location")

    graph.add_new_triplet("Vera", "is_a", "character", check_synonyms=False, head_feature="character", tail_feature="type")
    graph.add_new_triplet("Vera", "appears_in", "The Accompanist", check_synonyms=False, head_feature="character", tail_feature="book")
    graph.add_new_triplet("Vera", "loves", "Sofya", check_synonyms=False, head_feature="character", tail_feature="character")
    graph.add_new_triplet("Sofya", "is_a", "character", check_synonyms=False, head_feature="character", tail_feature="type")
    graph.add_new_triplet("Sofya", "appears_in", "The Accompanist", check_synonyms=False, head_feature="character", tail_feature="book")

    graph.add_new_triplet("Vera", "inspired_by", "Anna Karenina", check_synonyms=False, head_feature="character", tail_feature="character")
    graph.add_new_triplet("Nina Berberova", "influenced", "Lyudmila Ulitskaya", check_synonyms=False, head_feature="person", tail_feature="person")
    graph.add_new_triplet("The Accompanist", "referenced_by", "Sonechka", check_synonyms=False, head_feature="book", tail_feature="book")
    graph.add_new_triplet("Nina Berberova", "lived_in", "Saint_Petersburg", check_synonyms=False, head_feature="person", tail_feature="location")

    return graph


def create_test_graph() -> GraphData:
    graph = GraphData()

    graph.add_new_triplet("Leo Tolstoy", "is_a", "author", check_synonyms=False, head_feature="person",
                          tail_feature="type")
    graph.add_new_triplet("Fyodor Dostoevsky", "is_a", "author", check_synonyms=False, head_feature="person",
                          tail_feature="type")
    graph.add_new_triplet("Anton Chekhov", "is_a", "author", check_synonyms=False, head_feature="person",
                          tail_feature="type")
    graph.add_new_triplet("Alexander Pushkin", "is_a", "author", check_synonyms=False, head_feature="person",
                          tail_feature="type")
    graph.add_new_triplet("Nikolai Gogol", "is_a", "author", check_synonyms=False, head_feature="person",
                          tail_feature="type")
    graph.add_new_triplet("Ivan Turgenev", "is_a", "author", check_synonyms=False, head_feature="person",
                          tail_feature="type")
    graph.add_new_triplet("Mikhail Bulgakov", "is_a", "author", check_synonyms=False, head_feature="person",
                          tail_feature="type")
    graph.add_new_triplet("Vladimir Nabokov", "is_a", "author", check_synonyms=False, head_feature="person",
                          tail_feature="type")

    graph.add_new_triplet("War and Peace", "is_a", "novel", check_synonyms=False, head_feature="book",
                          tail_feature="type")
    graph.add_new_triplet("Anna Karenina", "is_a", "novel", check_synonyms=False, head_feature="book",
                          tail_feature="type")
    graph.add_new_triplet("Crime and Punishment", "is_a", "novel", check_synonyms=False, head_feature="book",
                          tail_feature="type")
    graph.add_new_triplet("The Cherry Orchard", "is_a", "play", check_synonyms=False, head_feature="book",
                          tail_feature="type")
    graph.add_new_triplet("The Brothers Karamazov", "is_a", "novel", check_synonyms=False, head_feature="book",
                          tail_feature="type")
    graph.add_new_triplet("Idiot", "is_a", "novel", check_synonyms=False, head_feature="book", tail_feature="type")
    graph.add_new_triplet("Eugene Onegin", "is_a", "poem", check_synonyms=False, head_feature="book",
                          tail_feature="type")
    graph.add_new_triplet("Dead Souls", "is_a", "novel", check_synonyms=False, head_feature="book", tail_feature="type")
    graph.add_new_triplet("Fathers and Sons", "is_a", "novel", check_synonyms=False, head_feature="book",
                          tail_feature="type")
    graph.add_new_triplet("The Master and Margarita", "is_a", "novel", check_synonyms=False, head_feature="book",
                          tail_feature="type")
    graph.add_new_triplet("Lolita", "is_a", "novel", check_synonyms=False, head_feature="book", tail_feature="type")
    graph.add_new_triplet("The Seagull", "is_a", "play", check_synonyms=False, head_feature="book", tail_feature="type")
    graph.add_new_triplet("Uncle Vanya", "is_a", "play", check_synonyms=False, head_feature="book", tail_feature="type")
    graph.add_new_triplet("The Queen of Spades", "is_a", "short_story", check_synonyms=False, head_feature="book",
                          tail_feature="type")
    graph.add_new_triplet("Notes from Underground", "is_a", "novella", check_synonyms=False, head_feature="book",
                          tail_feature="type")

    graph.add_new_triplet("Leo Tolstoy", "wrote", "War and Peace", check_synonyms=False, head_feature="person",
                          tail_feature="book")
    graph.add_new_triplet("Leo Tolstoy", "wrote", "Anna Karenina", check_synonyms=False, head_feature="person",
                          tail_feature="book")
    graph.add_new_triplet("Fyodor Dostoevsky", "wrote", "The Brothers Karamazov", check_synonyms=False,
                          head_feature="person", tail_feature="book")
    graph.add_new_triplet("Fyodor Dostoevsky", "wrote", "Idiot", check_synonyms=False, head_feature="person",
                          tail_feature="book")
    graph.add_new_triplet("Fyodor Dostoevsky", "wrote", "Crime and Punishment", check_synonyms=False,
                          head_feature="person", tail_feature="book")
    graph.add_new_triplet("Fyodor Dostoevsky", "wrote", "Notes from Underground", check_synonyms=False,
                          head_feature="person", tail_feature="book")
    graph.add_new_triplet("Anton Chekhov", "wrote", "The Cherry Orchard", check_synonyms=False, head_feature="person",
                          tail_feature="book")
    graph.add_new_triplet("Anton Chekhov", "wrote", "The Seagull", check_synonyms=False, head_feature="person",
                          tail_feature="book")
    graph.add_new_triplet("Anton Chekhov", "wrote", "Uncle Vanya", check_synonyms=False, head_feature="person",
                          tail_feature="book")
    graph.add_new_triplet("Alexander Pushkin", "wrote", "Eugene Onegin", check_synonyms=False, head_feature="person",
                          tail_feature="book")
    graph.add_new_triplet("Alexander Pushkin", "wrote", "The Queen of Spades", check_synonyms=False,
                          head_feature="person", tail_feature="book")
    graph.add_new_triplet("Nikolai Gogol", "wrote", "Dead Souls", check_synonyms=False, head_feature="person",
                          tail_feature="book")
    graph.add_new_triplet("Ivan Turgenev", "wrote", "Fathers and Sons", check_synonyms=False, head_feature="person",
                          tail_feature="book")
    graph.add_new_triplet("Mikhail Bulgakov", "wrote", "The Master and Margarita", check_synonyms=False,
                          head_feature="person", tail_feature="book")
    graph.add_new_triplet("Vladimir Nabokov", "wrote", "Lolita", check_synonyms=False, head_feature="person",
                          tail_feature="book")

    graph.add_new_triplet("Pierre Bezukhov", "is_a", "character", check_synonyms=False, head_feature="character",
                          tail_feature="type")
    graph.add_new_triplet("Anna Karenina", "is_a", "character", check_synonyms=False, head_feature="character",
                          tail_feature="type")
    graph.add_new_triplet("Raskolnikov", "is_a", "character", check_synonyms=False, head_feature="character",
                          tail_feature="type")
    graph.add_new_triplet("Ranevskaya", "is_a", "character", check_synonyms=False, head_feature="character",
                          tail_feature="type")
    graph.add_new_triplet("Alyosha Karamazov", "is_a", "character", check_synonyms=False, head_feature="character",
                          tail_feature="type")
    graph.add_new_triplet("Prince Myshkin", "is_a", "character", check_synonyms=False, head_feature="character",
                          tail_feature="type")
    graph.add_new_triplet("Eugene Onegin", "is_a", "character", check_synonyms=False, head_feature="character",
                          tail_feature="type")
    graph.add_new_triplet("Tatyana Larina", "is_a", "character", check_synonyms=False, head_feature="character",
                          tail_feature="type")
    graph.add_new_triplet("Chichikov", "is_a", "character", check_synonyms=False, head_feature="character",
                          tail_feature="type")
    graph.add_new_triplet("Bazarov", "is_a", "character", check_synonyms=False, head_feature="character",
                          tail_feature="type")
    graph.add_new_triplet("Woland", "is_a", "character", check_synonyms=False, head_feature="character",
                          tail_feature="type")
    graph.add_new_triplet("Margarita", "is_a", "character", check_synonyms=False, head_feature="character",
                          tail_feature="type")
    graph.add_new_triplet("Humbert Humbert", "is_a", "character", check_synonyms=False, head_feature="character",
                          tail_feature="type")
    graph.add_new_triplet("Natasha Rostova", "is_a", "character", check_synonyms=False, head_feature="character",
                          tail_feature="type")
    graph.add_new_triplet("Andrei Bolkonsky", "is_a", "character", check_synonyms=False, head_feature="character",
                          tail_feature="type")
    graph.add_new_triplet("Sonya Marmeladova", "is_a", "character", check_synonyms=False, head_feature="character",
                          tail_feature="type")
    graph.add_new_triplet("Nina Zarechnaya", "is_a", "character", check_synonyms=False, head_feature="character",
                          tail_feature="type")

    graph.add_new_triplet("Pierre Bezukhov", "appears_in", "War and Peace", check_synonyms=False,
                          head_feature="character", tail_feature="book")
    graph.add_new_triplet("Natasha Rostova", "appears_in", "War and Peace", check_synonyms=False,
                          head_feature="character", tail_feature="book")
    graph.add_new_triplet("Andrei Bolkonsky", "appears_in", "War and Peace", check_synonyms=False,
                          head_feature="character", tail_feature="book")
    graph.add_new_triplet("Anna Karenina", "appears_in", "Anna Karenina", check_synonyms=False,
                          head_feature="character", tail_feature="book")
    graph.add_new_triplet("Raskolnikov", "appears_in", "Crime and Punishment", check_synonyms=False,
                          head_feature="character", tail_feature="book")
    graph.add_new_triplet("Sonya Marmeladova", "appears_in", "Crime and Punishment", check_synonyms=False,
                          head_feature="character", tail_feature="book")
    graph.add_new_triplet("Ranevskaya", "appears_in", "The Cherry Orchard", check_synonyms=False,
                          head_feature="character", tail_feature="book")
    graph.add_new_triplet("Alyosha Karamazov", "appears_in", "The Brothers Karamazov", check_synonyms=False,
                          head_feature="character", tail_feature="book")
    graph.add_new_triplet("Prince Myshkin", "appears_in", "Idiot", check_synonyms=False, head_feature="character",
                          tail_feature="book")
    graph.add_new_triplet("Eugene Onegin", "appears_in", "Eugene Onegin", check_synonyms=False,
                          head_feature="character", tail_feature="book")
    graph.add_new_triplet("Tatyana Larina", "appears_in", "Eugene Onegin", check_synonyms=False,
                          head_feature="character", tail_feature="book")
    graph.add_new_triplet("Chichikov", "appears_in", "Dead Souls", check_synonyms=False, head_feature="character",
                          tail_feature="book")
    graph.add_new_triplet("Bazarov", "appears_in", "Fathers and Sons", check_synonyms=False, head_feature="character",
                          tail_feature="book")
    graph.add_new_triplet("Woland", "appears_in", "The Master and Margarita", check_synonyms=False,
                          head_feature="character", tail_feature="book")
    graph.add_new_triplet("Margarita", "appears_in", "The Master and Margarita", check_synonyms=False,
                          head_feature="character", tail_feature="book")
    graph.add_new_triplet("Humbert Humbert", "appears_in", "Lolita", check_synonyms=False, head_feature="character",
                          tail_feature="book")
    graph.add_new_triplet("Nina Zarechnaya", "appears_in", "The Seagull", check_synonyms=False,
                          head_feature="character", tail_feature="book")

    graph.add_new_triplet("Leo Tolstoy", "influenced", "Anton Chekhov", check_synonyms=False, head_feature="person",
                          tail_feature="person")
    graph.add_new_triplet("Fyodor Dostoevsky", "influenced", "Leo Tolstoy", check_synonyms=False, head_feature="person",
                          tail_feature="person")
    graph.add_new_triplet("Alexander Pushkin", "influenced", "Nikolai Gogol", check_synonyms=False,
                          head_feature="person", tail_feature="person")
    graph.add_new_triplet("Alexander Pushkin", "influenced", "Leo Tolstoy", check_synonyms=False, head_feature="person",
                          tail_feature="person")
    graph.add_new_triplet("Nikolai Gogol", "influenced", "Fyodor Dostoevsky", check_synonyms=False,
                          head_feature="person", tail_feature="person")
    graph.add_new_triplet("Ivan Turgenev", "influenced", "Anton Chekhov", check_synonyms=False, head_feature="person",
                          tail_feature="person")
    graph.add_new_triplet("Fyodor Dostoevsky", "influenced", "Mikhail Bulgakov", check_synonyms=False,
                          head_feature="person", tail_feature="person")
    graph.add_new_triplet("Nikolai Gogol", "influenced", "Mikhail Bulgakov", check_synonyms=False,
                          head_feature="person", tail_feature="person")
    graph.add_new_triplet("Vladimir Nabokov", "influenced_by", "Alexander Pushkin", check_synonyms=False,
                          head_feature="person", tail_feature="person")

    graph.add_new_triplet("War and Peace", "belongs_to_genre", "realism", check_synonyms=False, head_feature="book",
                          tail_feature="genre")
    graph.add_new_triplet("Anna Karenina", "belongs_to_genre", "realism", check_synonyms=False, head_feature="book",
                          tail_feature="genre")
    graph.add_new_triplet("Crime and Punishment", "belongs_to_genre", "psychological_fiction", check_synonyms=False,
                          head_feature="book", tail_feature="genre")
    graph.add_new_triplet("The Brothers Karamazov", "belongs_to_genre", "philosophical_novel", check_synonyms=False,
                          head_feature="book", tail_feature="genre")
    graph.add_new_triplet("The Cherry Orchard", "belongs_to_genre", "drama", check_synonyms=False, head_feature="book",
                          tail_feature="genre")
    graph.add_new_triplet("Eugene Onegin", "belongs_to_genre", "poetry", check_synonyms=False, head_feature="book",
                          tail_feature="genre")
    graph.add_new_triplet("Dead Souls", "belongs_to_genre", "satire", check_synonyms=False, head_feature="book",
                          tail_feature="genre")
    graph.add_new_triplet("Fathers and Sons", "belongs_to_genre", "realism", check_synonyms=False, head_feature="book",
                          tail_feature="genre")
    graph.add_new_triplet("The Master and Margarita", "belongs_to_genre", "magical_realism", check_synonyms=False,
                          head_feature="book", tail_feature="genre")
    graph.add_new_triplet("Lolita", "belongs_to_genre", "controversial_fiction", check_synonyms=False,
                          head_feature="book", tail_feature="genre")
    graph.add_new_triplet("The Seagull", "belongs_to_genre", "drama", check_synonyms=False, head_feature="book",
                          tail_feature="genre")
    graph.add_new_triplet("Uncle Vanya", "belongs_to_genre", "drama", check_synonyms=False, head_feature="book",
                          tail_feature="genre")
    graph.add_new_triplet("The Queen of Spades", "belongs_to_genre", "gothic_fiction", check_synonyms=False,
                          head_feature="book", tail_feature="genre")
    graph.add_new_triplet("Notes from Underground", "belongs_to_genre", "existentialism", check_synonyms=False,
                          head_feature="book", tail_feature="genre")

    graph.add_new_triplet("Leo Tolstoy", "lived_during", "19th_century", check_synonyms=False, head_feature="person",
                          tail_feature="time_period")
    graph.add_new_triplet("Fyodor Dostoevsky", "lived_during", "19th_century", check_synonyms=False,
                          head_feature="person", tail_feature="time_period")
    graph.add_new_triplet("Anton Chekhov", "lived_during", "19th_century", check_synonyms=False, head_feature="person",
                          tail_feature="time_period")
    graph.add_new_triplet("Alexander Pushkin", "lived_during", "19th_century", check_synonyms=False,
                          head_feature="person", tail_feature="time_period")
    graph.add_new_triplet("Nikolai Gogol", "lived_during", "19th_century", check_synonyms=False, head_feature="person",
                          tail_feature="time_period")
    graph.add_new_triplet("Ivan Turgenev", "lived_during", "19th_century", check_synonyms=False, head_feature="person",
                          tail_feature="time_period")
    graph.add_new_triplet("Mikhail Bulgakov", "lived_during", "20th_century", check_synonyms=False,
                          head_feature="person", tail_feature="time_period")
    graph.add_new_triplet("Vladimir Nabokov", "lived_during", "20th_century", check_synonyms=False,
                          head_feature="person", tail_feature="time_period")

    graph.add_new_triplet("Pierre Bezukhov", "loves", "Natasha Rostova", check_synonyms=False, head_feature="character",
                          tail_feature="character")
    graph.add_new_triplet("Andrei Bolkonsky", "loves", "Natasha Rostova", check_synonyms=False,
                          head_feature="character", tail_feature="character")
    graph.add_new_triplet("Eugene Onegin", "rejects", "Tatyana Larina", check_synonyms=False, head_feature="character",
                          tail_feature="character")
    graph.add_new_triplet("Raskolnikov", "confesses_to", "Sonya Marmeladova", check_synonyms=False,
                          head_feature="character", tail_feature="character")
    graph.add_new_triplet("Woland", "interacts_with", "Margarita", check_synonyms=False, head_feature="character",
                          tail_feature="character")

    graph.add_new_triplet("War and Peace", "explores_theme", "war", check_synonyms=False, head_feature="book",
                          tail_feature="theme")
    graph.add_new_triplet("War and Peace", "explores_theme", "peace", check_synonyms=False, head_feature="book",
                          tail_feature="theme")
    graph.add_new_triplet("Anna Karenina", "explores_theme", "love", check_synonyms=False, head_feature="book",
                          tail_feature="theme")
    graph.add_new_triplet("Anna Karenina", "explores_theme", "betrayal", check_synonyms=False, head_feature="book",
                          tail_feature="theme")
    graph.add_new_triplet("Crime and Punishment", "explores_theme", "guilt", check_synonyms=False, head_feature="book",
                          tail_feature="theme")
    graph.add_new_triplet("Crime and Punishment", "explores_theme", "redemption", check_synonyms=False,
                          head_feature="book", tail_feature="theme")
    graph.add_new_triplet("The Master and Margarita", "explores_theme", "good_and_evil", check_synonyms=False,
                          head_feature="book", tail_feature="theme")
    graph.add_new_triplet("Fathers and Sons", "explores_theme", "generational_conflict", check_synonyms=False,
                          head_feature="book", tail_feature="theme")
    graph.add_new_triplet("Dead Souls", "explores_theme", "corruption", check_synonyms=False, head_feature="book",
                          tail_feature="theme")

    graph.add_new_triplet("Alexander Solzhenitsyn", "is_a", "author", check_synonyms=False, head_feature="person",
                          tail_feature="type")
    graph.add_new_triplet("Boris Pasternak", "is_a", "author", check_synonyms=False, head_feature="person",
                          tail_feature="type")
    graph.add_new_triplet("Maxim Gorky", "is_a", "author", check_synonyms=False, head_feature="person",
                          tail_feature="type")
    graph.add_new_triplet("Marina Tsvetaeva", "is_a", "author", check_synonyms=False, head_feature="person",
                          tail_feature="type")
    graph.add_new_triplet("Anna Akhmatova", "is_a", "author", check_synonyms=False, head_feature="person",
                          tail_feature="type")
    graph.add_new_triplet("Sergei Yesenin", "is_a", "author", check_synonyms=False, head_feature="person",
                          tail_feature="type")
    graph.add_new_triplet("Isaac Babel", "is_a", "author", check_synonyms=False, head_feature="person",
                          tail_feature="type")

    graph.add_new_triplet("One Day in the Life of Ivan Denisovich", "is_a", "novel", check_synonyms=False,
                          head_feature="book", tail_feature="type")
    graph.add_new_triplet("The Gulag Archipelago", "is_a", "non_fiction", check_synonyms=False, head_feature="book",
                          tail_feature="type")
    graph.add_new_triplet("Doctor Zhivago", "is_a", "novel", check_synonyms=False, head_feature="book",
                          tail_feature="type")
    graph.add_new_triplet("Mother", "is_a", "novel", check_synonyms=False, head_feature="book", tail_feature="type")
    graph.add_new_triplet("The Lower Depths", "is_a", "play", check_synonyms=False, head_feature="book",
                          tail_feature="type")
    graph.add_new_triplet("Requiem", "is_a", "poem", check_synonyms=False, head_feature="book", tail_feature="type")
    graph.add_new_triplet("Pale Fire", "is_a", "novel", check_synonyms=False, head_feature="book", tail_feature="type")
    graph.add_new_triplet("Speak, Memory", "is_a", "memoir", check_synonyms=False, head_feature="book",
                          tail_feature="type")
    graph.add_new_triplet("Red Cavalry", "is_a", "short_story_collection", check_synonyms=False, head_feature="book",
                          tail_feature="type")

    graph.add_new_triplet("Alexander Solzhenitsyn", "wrote", "One Day in the Life of Ivan Denisovich",
                          check_synonyms=False, head_feature="person", tail_feature="book")
    graph.add_new_triplet("Alexander Solzhenitsyn", "wrote", "The Gulag Archipelago", check_synonyms=False,
                          head_feature="person", tail_feature="book")
    graph.add_new_triplet("Boris Pasternak", "wrote", "Doctor Zhivago", check_synonyms=False, head_feature="person",
                          tail_feature="book")
    graph.add_new_triplet("Maxim Gorky", "wrote", "Mother", check_synonyms=False, head_feature="person",
                          tail_feature="book")
    graph.add_new_triplet("Maxim Gorky", "wrote", "The Lower Depths", check_synonyms=False, head_feature="person",
                          tail_feature="book")
    graph.add_new_triplet("Anna Akhmatova", "wrote", "Requiem", check_synonyms=False, head_feature="person",
                          tail_feature="book")
    graph.add_new_triplet("Vladimir Nabokov", "wrote", "Pale Fire", check_synonyms=False, head_feature="person",
                          tail_feature="book")
    graph.add_new_triplet("Vladimir Nabokov", "wrote", "Speak, Memory", check_synonyms=False, head_feature="person",
                          tail_feature="book")
    graph.add_new_triplet("Isaac Babel", "wrote", "Red Cavalry", check_synonyms=False, head_feature="person",
                          tail_feature="book")

    graph.add_new_triplet("Ivan Denisovich", "is_a", "character", check_synonyms=False, head_feature="character",
                          tail_feature="type")
    graph.add_new_triplet("Yuri Zhivago", "is_a", "character", check_synonyms=False, head_feature="character",
                          tail_feature="type")
    graph.add_new_triplet("Lara Antipova", "is_a", "character", check_synonyms=False, head_feature="character",
                          tail_feature="type")
    graph.add_new_triplet("Pavel Vlasov", "is_a", "character", check_synonyms=False, head_feature="character",
                          tail_feature="type")
    graph.add_new_triplet("Nilovna", "is_a", "character", check_synonyms=False, head_feature="character",
                          tail_feature="type")
    graph.add_new_triplet("Luka", "is_a", "character", check_synonyms=False, head_feature="character",
                          tail_feature="type")
    graph.add_new_triplet("Charles Kinbote", "is_a", "character", check_synonyms=False, head_feature="character",
                          tail_feature="type")
    graph.add_new_triplet("John Shade", "is_a", "character", check_synonyms=False, head_feature="character",
                          tail_feature="type")
    graph.add_new_triplet("Lyutov", "is_a", "character", check_synonyms=False, head_feature="character",
                          tail_feature="type")

    graph.add_new_triplet("Ivan Denisovich", "appears_in", "One Day in the Life of Ivan Denisovich",
                          check_synonyms=False, head_feature="character", tail_feature="book")
    graph.add_new_triplet("Yuri Zhivago", "appears_in", "Doctor Zhivago", check_synonyms=False,
                          head_feature="character", tail_feature="book")
    graph.add_new_triplet("Lara Antipova", "appears_in", "Doctor Zhivago", check_synonyms=False,
                          head_feature="character", tail_feature="book")
    graph.add_new_triplet("Pavel Vlasov", "appears_in", "Mother", check_synonyms=False, head_feature="character",
                          tail_feature="book")
    graph.add_new_triplet("Nilovna", "appears_in", "Mother", check_synonyms=False, head_feature="character",
                          tail_feature="book")
    graph.add_new_triplet("Luka", "appears_in", "The Lower Depths", check_synonyms=False, head_feature="character",
                          tail_feature="book")
    graph.add_new_triplet("Charles Kinbote", "appears_in", "Pale Fire", check_synonyms=False, head_feature="character",
                          tail_feature="book")
    graph.add_new_triplet("John Shade", "appears_in", "Pale Fire", check_synonyms=False, head_feature="character",
                          tail_feature="book")
    graph.add_new_triplet("Lyutov", "appears_in", "Red Cavalry", check_synonyms=False, head_feature="character",
                          tail_feature="book")

    graph.add_new_triplet("Leo Tolstoy", "influenced", "Boris Pasternak", check_synonyms=False, head_feature="person",
                          tail_feature="person")
    graph.add_new_triplet("Fyodor Dostoevsky", "influenced", "Alexander Solzhenitsyn", check_synonyms=False,
                          head_feature="person", tail_feature="person")
    graph.add_new_triplet("Anton Chekhov", "influenced", "Isaac Babel", check_synonyms=False, head_feature="person",
                          tail_feature="person")
    graph.add_new_triplet("Maxim Gorky", "influenced", "Alexander Solzhenitsyn", check_synonyms=False,
                          head_feature="person", tail_feature="person")
    graph.add_new_triplet("Alexander Pushkin", "influenced", "Anna Akhmatova", check_synonyms=False,
                          head_feature="person", tail_feature="person")
    graph.add_new_triplet("Alexander Pushkin", "influenced", "Marina Tsvetaeva", check_synonyms=False,
                          head_feature="person", tail_feature="person")
    graph.add_new_triplet("Marina Tsvetaeva", "influenced", "Anna Akhmatova", check_synonyms=False,
                          head_feature="person", tail_feature="person")
    graph.add_new_triplet("Anna Akhmatova", "influenced", "Sergei Yesenin", check_synonyms=False, head_feature="person",
                          tail_feature="person")

    graph.add_new_triplet("One Day in the Life of Ivan Denisovich", "belongs_to_genre", "historical_fiction",
                          check_synonyms=False, head_feature="book", tail_feature="genre")
    graph.add_new_triplet("The Gulag Archipelago", "belongs_to_genre", "history", check_synonyms=False,
                          head_feature="book", tail_feature="genre")
    graph.add_new_triplet("Doctor Zhivago", "belongs_to_genre", "historical_fiction", check_synonyms=False,
                          head_feature="book", tail_feature="genre")
    graph.add_new_triplet("Mother", "belongs_to_genre", "social_realism", check_synonyms=False, head_feature="book",
                          tail_feature="genre")
    graph.add_new_triplet("The Lower Depths", "belongs_to_genre", "drama", check_synonyms=False, head_feature="book",
                          tail_feature="genre")
    graph.add_new_triplet("Requiem", "belongs_to_genre", "poetry", check_synonyms=False, head_feature="book",
                          tail_feature="genre")
    graph.add_new_triplet("Pale Fire", "belongs_to_genre", "postmodern", check_synonyms=False, head_feature="book",
                          tail_feature="genre")
    graph.add_new_triplet("Speak, Memory", "belongs_to_genre", "autobiography", check_synonyms=False,
                          head_feature="book", tail_feature="genre")
    graph.add_new_triplet("Red Cavalry", "belongs_to_genre", "war_fiction", check_synonyms=False, head_feature="book",
                          tail_feature="genre")

    graph.add_new_triplet("Alexander Solzhenitsyn", "lived_during", "20th_century", check_synonyms=False,
                          head_feature="person", tail_feature="time_period")
    graph.add_new_triplet("Boris Pasternak", "lived_during", "20th_century", check_synonyms=False,
                          head_feature="person", tail_feature="time_period")
    graph.add_new_triplet("Maxim Gorky", "lived_during", "19th_century", check_synonyms=False, head_feature="person",
                          tail_feature="time_period")
    graph.add_new_triplet("Maxim Gorky", "lived_during", "20th_century", check_synonyms=False, head_feature="person",
                          tail_feature="time_period")
    graph.add_new_triplet("Marina Tsvetaeva", "lived_during", "20th_century", check_synonyms=False,
                          head_feature="person", tail_feature="time_period")
    graph.add_new_triplet("Anna Akhmatova", "lived_during", "20th_century", check_synonyms=False, head_feature="person",
                          tail_feature="time_period")
    graph.add_new_triplet("Sergei Yesenin", "lived_during", "20th_century", check_synonyms=False, head_feature="person",
                          tail_feature="time_period")
    graph.add_new_triplet("Isaac Babel", "lived_during", "20th_century", check_synonyms=False, head_feature="person",
                          tail_feature="time_period")

    graph.add_new_triplet("Alexander Solzhenitsyn", "won_award", "Nobel_Prize_in_Literature", check_synonyms=False,
                          head_feature="person", tail_feature="award")
    graph.add_new_triplet("Boris Pasternak", "won_award", "Nobel_Prize_in_Literature", check_synonyms=False,
                          head_feature="person", tail_feature="award")
    graph.add_new_triplet("Ivan Bunin", "is_a", "author", check_synonyms=False, head_feature="person",
                          tail_feature="type")
    graph.add_new_triplet("Ivan Bunin", "won_award", "Nobel_Prize_in_Literature", check_synonyms=False,
                          head_feature="person", tail_feature="award")
    graph.add_new_triplet("Joseph Brodsky", "is_a", "author", check_synonyms=False, head_feature="person",
                          tail_feature="type")
    graph.add_new_triplet("Joseph Brodsky", "won_award", "Nobel_Prize_in_Literature", check_synonyms=False,
                          head_feature="person", tail_feature="award")

    graph.add_new_triplet("Symbolism", "is_a", "literary_movement", check_synonyms=False, head_feature="movement",
                          tail_feature="type")
    graph.add_new_triplet("Socialist_Realism", "is_a", "literary_movement", check_synonyms=False,
                          head_feature="movement", tail_feature="type")
    graph.add_new_triplet("Acmeism", "is_a", "literary_movement", check_synonyms=False, head_feature="movement",
                          tail_feature="type")
    graph.add_new_triplet("Futurism", "is_a", "literary_movement", check_synonyms=False, head_feature="movement",
                          tail_feature="type")

    graph.add_new_triplet("Alexander Blok", "is_a", "author", check_synonyms=False, head_feature="person",
                          tail_feature="type")
    graph.add_new_triplet("Andrei Bely", "is_a", "author", check_synonyms=False, head_feature="person",
                          tail_feature="type")
    graph.add_new_triplet("Alexander Blok", "associated_with", "Symbolism", check_synonyms=False, head_feature="person",
                          tail_feature="movement")
    graph.add_new_triplet("Andrei Bely", "associated_with", "Symbolism", check_synonyms=False, head_feature="person",
                          tail_feature="movement")
    graph.add_new_triplet("Maxim Gorky", "associated_with", "Socialist_Realism", check_synonyms=False,
                          head_feature="person", tail_feature="movement")
    graph.add_new_triplet("Anna Akhmatova", "associated_with", "Acmeism", check_synonyms=False, head_feature="person",
                          tail_feature="movement")
    graph.add_new_triplet("Vladimir Mayakovsky", "is_a", "author", check_synonyms=False, head_feature="person",
                          tail_feature="type")
    graph.add_new_triplet("Vladimir Mayakovsky", "associated_with", "Futurism", check_synonyms=False,
                          head_feature="person", tail_feature="movement")

    graph.add_new_triplet("Dark Avenues", "is_a", "short_story_collection", check_synonyms=False, head_feature="book",
                          tail_feature="type")
    graph.add_new_triplet("Ivan Bunin", "wrote", "Dark Avenues", check_synonyms=False, head_feature="person",
                          tail_feature="book")
    graph.add_new_triplet("Dark Avenues", "belongs_to_genre", "realism", check_synonyms=False, head_feature="book",
                          tail_feature="genre")

    graph.add_new_triplet("A Part of Speech", "is_a", "poetry_collection", check_synonyms=False, head_feature="book",
                          tail_feature="type")
    graph.add_new_triplet("Joseph Brodsky", "wrote", "A Part of Speech", check_synonyms=False, head_feature="person",
                          tail_feature="book")
    graph.add_new_triplet("A Part of Speech", "belongs_to_genre", "poetry", check_synonyms=False, head_feature="book",
                          tail_feature="genre")

    graph.add_new_triplet("The Twelve", "is_a", "poem", check_synonyms=False, head_feature="book", tail_feature="type")
    graph.add_new_triplet("Alexander Blok", "wrote", "The Twelve", check_synonyms=False, head_feature="person",
                          tail_feature="book")
    graph.add_new_triplet("The Twelve", "belongs_to_genre", "poetry", check_synonyms=False, head_feature="book",
                          tail_feature="genre")

    graph.add_new_triplet("Petersburg", "is_a", "novel", check_synonyms=False, head_feature="book", tail_feature="type")
    graph.add_new_triplet("Andrei Bely", "wrote", "Petersburg", check_synonyms=False, head_feature="person",
                          tail_feature="book")
    graph.add_new_triplet("Petersburg", "belongs_to_genre", "symbolist_fiction", check_synonyms=False,
                          head_feature="book", tail_feature="genre")

    graph.add_new_triplet("A Cloud in Trousers", "is_a", "poem", check_synonyms=False, head_feature="book",
                          tail_feature="type")
    graph.add_new_triplet("Vladimir Mayakovsky", "wrote", "A Cloud in Trousers", check_synonyms=False,
                          head_feature="person", tail_feature="book")
    graph.add_new_triplet("A Cloud in Trousers", "belongs_to_genre", "futurist_poetry", check_synonyms=False,
                          head_feature="book", tail_feature="genre")

    graph.add_new_triplet("Richard Pevear", "is_a", "translator", check_synonyms=False, head_feature="person",
                          tail_feature="type")
    graph.add_new_triplet("Larissa Volokhonsky", "is_a", "translator", check_synonyms=False, head_feature="person",
                          tail_feature="type")
    graph.add_new_triplet("Constance Garnett", "is_a", "translator", check_synonyms=False, head_feature="person",
                          tail_feature="type")

    graph.add_new_triplet("Richard Pevear", "translated", "War and Peace", check_synonyms=False, head_feature="person",
                          tail_feature="book")
    graph.add_new_triplet("Larissa Volokhonsky", "translated", "War and Peace", check_synonyms=False,
                          head_feature="person", tail_feature="book")
    graph.add_new_triplet("Richard Pevear", "translated", "Anna Karenina", check_synonyms=False, head_feature="person",
                          tail_feature="book")
    graph.add_new_triplet("Larissa Volokhonsky", "translated", "Anna Karenina", check_synonyms=False,
                          head_feature="person", tail_feature="book")
    graph.add_new_triplet("Constance Garnett", "translated", "Crime and Punishment", check_synonyms=False,
                          head_feature="person", tail_feature="book")
    graph.add_new_triplet("Constance Garnett", "translated", "The Brothers Karamazov", check_synonyms=False,
                          head_feature="person", tail_feature="book")

    graph.add_new_triplet("Moscow", "is_a", "location", check_synonyms=False, head_feature="location",
                          tail_feature="type")
    graph.add_new_triplet("Saint_Petersburg", "is_a", "location", check_synonyms=False, head_feature="location",
                          tail_feature="type")
    graph.add_new_triplet("Yasnaya_Polyana", "is_a", "location", check_synonyms=False, head_feature="location",
                          tail_feature="type")

    graph.add_new_triplet("War and Peace", "set_in", "Moscow", check_synonyms=False, head_feature="book",
                          tail_feature="location")
    graph.add_new_triplet("Crime and Punishment", "set_in", "Saint_Petersburg", check_synonyms=False,
                          head_feature="book", tail_feature="location")
    graph.add_new_triplet("The Brothers Karamazov", "set_in", "Russia", check_synonyms=False, head_feature="book",
                          tail_feature="location")
    graph.add_new_triplet("Leo Tolstoy", "lived_in", "Yasnaya_Polyana", check_synonyms=False, head_feature="person",
                          tail_feature="location")
    graph.add_new_triplet("Fyodor Dostoevsky", "lived_in", "Saint_Petersburg", check_synonyms=False,
                          head_feature="person", tail_feature="location")

    graph.add_new_triplet("Pierre Bezukhov", "married_to", "Natasha Rostova", check_synonyms=False,
                          head_feature="character", tail_feature="character")
    graph.add_new_triplet("Yuri Zhivago", "loved", "Lara Antipova", check_synonyms=False, head_feature="character",
                          tail_feature="character")
    graph.add_new_triplet("Eugene Onegin", "rejected", "Tatyana Larina", check_synonyms=False, head_feature="character",
                          tail_feature="character")
    graph.add_new_triplet("Raskolnikov", "confessed_to", "Sonya Marmeladova", check_synonyms=False,
                          head_feature="character", tail_feature="character")

    graph.add_new_triplet("Raskolnikov", "confessed_to", "Sonya Marmeladova", check_synonyms=False,
                          head_feature="character", tail_feature="character")

    graph.add_new_triplet("Mikhail Lermontov", "is_a", "author", check_synonyms=False, head_feature="person",
                          tail_feature="type")
    graph.add_new_triplet("Mikhail Lermontov", "lived_during", "19th_century", check_synonyms=False,
                          head_feature="person", tail_feature="time_period")
    graph.add_new_triplet("Mikhail Lermontov", "influenced_by", "Alexander Pushkin", check_synonyms=False,
                          head_feature="person", tail_feature="person")
    graph.add_new_triplet("Mikhail Lermontov", "associated_with", "Romanticism", check_synonyms=False,
                          head_feature="person", tail_feature="movement")

    graph.add_new_triplet("Mikhail Sholokhov", "is_a", "author", check_synonyms=False, head_feature="person",
                          tail_feature="type")
    graph.add_new_triplet("Mikhail Sholokhov", "lived_during", "20th_century", check_synonyms=False,
                          head_feature="person", tail_feature="time_period")
    graph.add_new_triplet("Mikhail Sholokhov", "won_award", "Nobel_Prize_in_Literature", check_synonyms=False,
                          head_feature="person", tail_feature="award")

    graph.add_new_triplet("Ilya Ilf", "is_a", "author", check_synonyms=False, head_feature="person",
                          tail_feature="type")
    graph.add_new_triplet("Yevgeny Petrov", "is_a", "author", check_synonyms=False, head_feature="person",
                          tail_feature="type")
    graph.add_new_triplet("Ilya Ilf", "lived_during", "20th_century", check_synonyms=False, head_feature="person",
                          tail_feature="time_period")
    graph.add_new_triplet("Yevgeny Petrov", "lived_during", "20th_century", check_synonyms=False, head_feature="person",
                          tail_feature="time_period")

    graph.add_new_triplet("Venedikt Yerofeyev", "is_a", "author", check_synonyms=False, head_feature="person",
                          tail_feature="type")
    graph.add_new_triplet("Venedikt Yerofeyev", "lived_during", "20th_century", check_synonyms=False,
                          head_feature="person", tail_feature="time_period")

    graph.add_new_triplet("Mikhail Saltykov-Shchedrin", "is_a", "author", check_synonyms=False, head_feature="person",
                          tail_feature="type")
    graph.add_new_triplet("Mikhail Saltykov-Shchedrin", "lived_during", "19th_century", check_synonyms=False,
                          head_feature="person", tail_feature="time_period")

    graph.add_new_triplet("Ivan Goncharov", "is_a", "author", check_synonyms=False, head_feature="person",
                          tail_feature="type")
    graph.add_new_triplet("Ivan Goncharov", "lived_during", "19th_century", check_synonyms=False, head_feature="person",
                          tail_feature="time_period")

    graph.add_new_triplet("Andrei Platonov", "is_a", "author", check_synonyms=False, head_feature="person",
                          tail_feature="type")
    graph.add_new_triplet("Andrei Platonov", "lived_during", "20th_century", check_synonyms=False,
                          head_feature="person", tail_feature="time_period")

    graph.add_new_triplet("Yevgeny Zamyatin", "is_a", "author", check_synonyms=False, head_feature="person",
                          tail_feature="type")
    graph.add_new_triplet("Yevgeny Zamyatin", "lived_during", "20th_century", check_synonyms=False,
                          head_feature="person", tail_feature="time_period")

    graph.add_new_triplet("Viktor Pelevin", "is_a", "author", check_synonyms=False, head_feature="person",
                          tail_feature="type")
    graph.add_new_triplet("Viktor Pelevin", "lived_during", "20th_century", check_synonyms=False, head_feature="person",
                          tail_feature="time_period")
    graph.add_new_triplet("Viktor Pelevin", "lived_during", "21st_century", check_synonyms=False, head_feature="person",
                          tail_feature="time_period")

    graph.add_new_triplet("A Hero of Our Time", "is_a", "novel", check_synonyms=False, head_feature="book",
                          tail_feature="type")
    graph.add_new_triplet("Mikhail Lermontov", "wrote", "A Hero of Our Time", check_synonyms=False,
                          head_feature="person", tail_feature="book")
    graph.add_new_triplet("A Hero of Our Time", "belongs_to_genre", "psychological_fiction", check_synonyms=False,
                          head_feature="book", tail_feature="genre")

    graph.add_new_triplet("And Quiet Flows the Don", "is_a", "novel", check_synonyms=False, head_feature="book",
                          tail_feature="type")
    graph.add_new_triplet("Mikhail Sholokhov", "wrote", "And Quiet Flows the Don", check_synonyms=False,
                          head_feature="person", tail_feature="book")
    graph.add_new_triplet("And Quiet Flows the Don", "belongs_to_genre", "historical_fiction", check_synonyms=False,
                          head_feature="book", tail_feature="genre")

    graph.add_new_triplet("The Twelve Chairs", "is_a", "novel", check_synonyms=False, head_feature="book",
                          tail_feature="type")
    graph.add_new_triplet("Ilya Ilf", "wrote", "The Twelve Chairs", check_synonyms=False, head_feature="person",
                          tail_feature="book")
    graph.add_new_triplet("Yevgeny Petrov", "wrote", "The Twelve Chairs", check_synonyms=False, head_feature="person",
                          tail_feature="book")
    graph.add_new_triplet("The Twelve Chairs", "belongs_to_genre", "satire", check_synonyms=False, head_feature="book",
                          tail_feature="genre")

    graph.add_new_triplet("The Golden Calf", "is_a", "novel", check_synonyms=False, head_feature="book",
                          tail_feature="type")
    graph.add_new_triplet("Ilya Ilf", "wrote", "The Golden Calf", check_synonyms=False, head_feature="person",
                          tail_feature="book")
    graph.add_new_triplet("Yevgeny Petrov", "wrote", "The Golden Calf", check_synonyms=False, head_feature="person",
                          tail_feature="book")
    graph.add_new_triplet("The Golden Calf", "belongs_to_genre", "satire", check_synonyms=False, head_feature="book",
                          tail_feature="genre")

    graph.add_new_triplet("Moscow to the End of the Line", "is_a", "novel", check_synonyms=False, head_feature="book",
                          tail_feature="type")
    graph.add_new_triplet("Venedikt Yerofeyev", "wrote", "Moscow to the End of the Line", check_synonyms=False,
                          head_feature="person", tail_feature="book")
    graph.add_new_triplet("Moscow to the End of the Line", "belongs_to_genre", "postmodern", check_synonyms=False,
                          head_feature="book", tail_feature="genre")

    graph.add_new_triplet("The History of a Town", "is_a", "novel", check_synonyms=False, head_feature="book",
                          tail_feature="type")
    graph.add_new_triplet("Mikhail Saltykov-Shchedrin", "wrote", "The History of a Town", check_synonyms=False,
                          head_feature="person", tail_feature="book")
    graph.add_new_triplet("The History of a Town", "belongs_to_genre", "satire", check_synonyms=False,
                          head_feature="book", tail_feature="genre")

    graph.add_new_triplet("Oblomov", "is_a", "novel", check_synonyms=False, head_feature="book", tail_feature="type")
    graph.add_new_triplet("Ivan Goncharov", "wrote", "Oblomov", check_synonyms=False, head_feature="person",
                          tail_feature="book")
    graph.add_new_triplet("Oblomov", "belongs_to_genre", "realism", check_synonyms=False, head_feature="book",
                          tail_feature="genre")

    graph.add_new_triplet("The Foundation Pit", "is_a", "novel", check_synonyms=False, head_feature="book",
                          tail_feature="type")
    graph.add_new_triplet("Andrei Platonov", "wrote", "The Foundation Pit", check_synonyms=False, head_feature="person",
                          tail_feature="book")
    graph.add_new_triplet("The Foundation Pit", "belongs_to_genre", "dystopian", check_synonyms=False,
                          head_feature="book", tail_feature="genre")

    graph.add_new_triplet("We", "is_a", "novel", check_synonyms=False, head_feature="book", tail_feature="type")
    graph.add_new_triplet("Yevgeny Zamyatin", "wrote", "We", check_synonyms=False, head_feature="person",
                          tail_feature="book")
    graph.add_new_triplet("We", "belongs_to_genre", "dystopian", check_synonyms=False, head_feature="book",
                          tail_feature="genre")

    graph.add_new_triplet("Generation P", "is_a", "novel", check_synonyms=False, head_feature="book",
                          tail_feature="type")
    graph.add_new_triplet("Viktor Pelevin", "wrote", "Generation P", check_synonyms=False, head_feature="person",
                          tail_feature="book")
    graph.add_new_triplet("Generation P", "belongs_to_genre", "postmodern", check_synonyms=False, head_feature="book",
                          tail_feature="genre")

    graph.add_new_triplet("Omon Ra", "is_a", "novel", check_synonyms=False, head_feature="book", tail_feature="type")
    graph.add_new_triplet("Viktor Pelevin", "wrote", "Omon Ra", check_synonyms=False, head_feature="person",
                          tail_feature="book")
    graph.add_new_triplet("Omon Ra", "belongs_to_genre", "satire", check_synonyms=False, head_feature="book",
                          tail_feature="genre")

    graph.add_new_triplet("Pechorin", "is_a", "character", check_synonyms=False, head_feature="character",
                          tail_feature="type")
    graph.add_new_triplet("Pechorin", "appears_in", "A Hero of Our Time", check_synonyms=False,
                          head_feature="character", tail_feature="book")

    graph.add_new_triplet("Grigory Melekhov", "is_a", "character", check_synonyms=False, head_feature="character",
                          tail_feature="type")
    graph.add_new_triplet("Grigory Melekhov", "appears_in", "And Quiet Flows the Don", check_synonyms=False,
                          head_feature="character", tail_feature="book")

    graph.add_new_triplet("Aksinia Astakhova", "is_a", "character", check_synonyms=False, head_feature="character",
                          tail_feature="type")
    graph.add_new_triplet("Aksinia Astakhova", "appears_in", "And Quiet Flows the Don", check_synonyms=False,
                          head_feature="character", tail_feature="book")

    graph.add_new_triplet("Ostap Bender", "is_a", "character", check_synonyms=False, head_feature="character",
                          tail_feature="type")
    graph.add_new_triplet("Ostap Bender", "appears_in", "The Twelve Chairs", check_synonyms=False,
                          head_feature="character", tail_feature="book")
    graph.add_new_triplet("Ostap Bender", "appears_in", "The Golden Calf", check_synonyms=False,
                          head_feature="character", tail_feature="book")

    graph.add_new_triplet("Kisa Vorobyaninov", "is_a", "character", check_synonyms=False, head_feature="character",
                          tail_feature="type")
    graph.add_new_triplet("Kisa Vorobyaninov", "appears_in", "The Twelve Chairs", check_synonyms=False,
                          head_feature="character", tail_feature="book")

    graph.add_new_triplet("Venichka", "is_a", "character", check_synonyms=False, head_feature="character",
                          tail_feature="type")
    graph.add_new_triplet("Venichka", "appears_in", "Moscow to the End of the Line", check_synonyms=False,
                          head_feature="character", tail_feature="book")

    graph.add_new_triplet("Ilya Oblomov", "is_a", "character", check_synonyms=False, head_feature="character",
                          tail_feature="type")
    graph.add_new_triplet("Ilya Oblomov", "appears_in", "Oblomov", check_synonyms=False, head_feature="character",
                          tail_feature="book")

    graph.add_new_triplet("Stolz", "is_a", "character", check_synonyms=False, head_feature="character",
                          tail_feature="type")
    graph.add_new_triplet("Stolz", "appears_in", "Oblomov", check_synonyms=False, head_feature="character",
                          tail_feature="book")

    graph.add_new_triplet("Voshchev", "is_a", "character", check_synonyms=False, head_feature="character",
                          tail_feature="type")
    graph.add_new_triplet("Voshchev", "appears_in", "The Foundation Pit", check_synonyms=False,
                          head_feature="character", tail_feature="book")

    graph.add_new_triplet("D-503", "is_a", "character", check_synonyms=False, head_feature="character",
                          tail_feature="type")
    graph.add_new_triplet("D-503", "appears_in", "We", check_synonyms=False, head_feature="character",
                          tail_feature="book")

    graph.add_new_triplet("I-330", "is_a", "character", check_synonyms=False, head_feature="character",
                          tail_feature="type")
    graph.add_new_triplet("I-330", "appears_in", "We", check_synonyms=False, head_feature="character",
                          tail_feature="book")

    graph.add_new_triplet("Tatarsky", "is_a", "character", check_synonyms=False, head_feature="character",
                          tail_feature="type")
    graph.add_new_triplet("Tatarsky", "appears_in", "Generation P", check_synonyms=False, head_feature="character",
                          tail_feature="book")

    graph.add_new_triplet("Omon", "is_a", "character", check_synonyms=False, head_feature="character",
                          tail_feature="type")
    graph.add_new_triplet("Omon", "appears_in", "Omon Ra", check_synonyms=False, head_feature="character",
                          tail_feature="book")

    graph.add_new_triplet("Caucasus", "is_a", "location", check_synonyms=False, head_feature="location",
                          tail_feature="type")
    graph.add_new_triplet("A Hero of Our Time", "set_in", "Caucasus", check_synonyms=False, head_feature="book",
                          tail_feature="location")

    graph.add_new_triplet("Don_River", "is_a", "location", check_synonyms=False, head_feature="location",
                          tail_feature="type")
    graph.add_new_triplet("And Quiet Flows the Don", "set_in", "Don_River", check_synonyms=False, head_feature="book",
                          tail_feature="location")

    graph.add_new_triplet("Odessa", "is_a", "location", check_synonyms=False, head_feature="location",
                          tail_feature="type")
    graph.add_new_triplet("The Twelve Chairs", "set_in", "Odessa", check_synonyms=False, head_feature="book",
                          tail_feature="location")

    graph.add_new_triplet("Petushki", "is_a", "location", check_synonyms=False, head_feature="location",
                          tail_feature="type")
    graph.add_new_triplet("Moscow to the End of the Line", "set_in", "Petushki", check_synonyms=False,
                          head_feature="book", tail_feature="location")

    graph.add_new_triplet("Glupov", "is_a", "location", check_synonyms=False, head_feature="location",
                          tail_feature="type")
    graph.add_new_triplet("The History of a Town", "set_in", "Glupov", check_synonyms=False, head_feature="book",
                          tail_feature="location")

    graph.add_new_triplet("OneState", "is_a", "location", check_synonyms=False, head_feature="location",
                          tail_feature="type")
    graph.add_new_triplet("We", "set_in", "OneState", check_synonyms=False, head_feature="book",
                          tail_feature="location")

    graph.add_new_triplet("Mikhail Lermontov", "influenced", "Fyodor Dostoevsky", check_synonyms=False,
                          head_feature="person", tail_feature="person")
    graph.add_new_triplet("Mikhail Lermontov", "influenced", "Leo Tolstoy", check_synonyms=False, head_feature="person",
                          tail_feature="person")
    graph.add_new_triplet("Ivan Goncharov", "influenced", "Anton Chekhov", check_synonyms=False, head_feature="person",
                          tail_feature="person")
    graph.add_new_triplet("Yevgeny Zamyatin", "influenced", "George Orwell", check_synonyms=False,
                          head_feature="person", tail_feature="person")
    graph.add_new_triplet("Yevgeny Zamyatin", "influenced", "Aldous Huxley", check_synonyms=False,
                          head_feature="person", tail_feature="person")
    graph.add_new_triplet("Mikhail Bulgakov", "influenced", "Viktor Pelevin", check_synonyms=False,
                          head_feature="person", tail_feature="person")
    graph.add_new_triplet("Vladimir Nabokov", "influenced", "Viktor Pelevin", check_synonyms=False,
                          head_feature="person", tail_feature="person")

    graph.add_new_triplet("Grigory Melekhov", "loved", "Aksinia Astakhova", check_synonyms=False,
                          head_feature="character", tail_feature="character")
    graph.add_new_triplet("Ilya Oblomov", "friends_with", "Stolz", check_synonyms=False, head_feature="character",
                          tail_feature="character")
    graph.add_new_triplet("D-503", "loved", "I-330", check_synonyms=False, head_feature="character",
                          tail_feature="character")

    graph.add_new_triplet("Leo Tolstoy", "lived_during", "20th_century", check_synonyms=False, head_feature="person",
                          tail_feature="time_period")
    graph.add_new_triplet("Anton Chekhov", "lived_during", "20th_century", check_synonyms=False, head_feature="person",
                          tail_feature="time_period")

    graph.add_new_triplet("Crime and Punishment", "belongs_to_genre", "philosophical_novel", check_synonyms=False,
                          head_feature="book", tail_feature="genre")
    graph.add_new_triplet("War and Peace", "belongs_to_genre", "historical_fiction", check_synonyms=False,
                          head_feature="book", tail_feature="genre")
    graph.add_new_triplet("Anna Karenina", "belongs_to_genre", "tragedy", check_synonyms=False, head_feature="book",
                          tail_feature="genre")

    graph.add_new_triplet("Mikhail Lermontov", "lived_in", "Moscow", check_synonyms=False, head_feature="person",
                          tail_feature="location")
    graph.add_new_triplet("Mikhail Lermontov", "lived_in", "Caucasus", check_synonyms=False, head_feature="person",
                          tail_feature="location")
    graph.add_new_triplet("Ivan Goncharov", "lived_in", "Saint_Petersburg", check_synonyms=False, head_feature="person",
                          tail_feature="location")
    graph.add_new_triplet("Yevgeny Zamyatin", "lived_in", "Saint_Petersburg", check_synonyms=False,
                          head_feature="person", tail_feature="location")
    graph.add_new_triplet("Viktor Pelevin", "lived_in", "Moscow", check_synonyms=False, head_feature="person",
                          tail_feature="location")

    graph.add_new_triplet("Boris Pasternak", "is_a", "author", check_synonyms=False, head_feature="person",
                          tail_feature="type")
    graph.add_new_triplet("Vladimir Mayakovsky", "is_a", "author", check_synonyms=False, head_feature="person",
                          tail_feature="type")
    graph.add_new_triplet("Doctor Zhivago", "is_a", "novel", check_synonyms=False, head_feature="book",
                          tail_feature="type")
    graph.add_new_triplet("Cloud in Pants", "is_a", "poem", check_synonyms=False, head_feature="book",
                          tail_feature="type")
    graph.add_new_triplet("Boris Pasternak", "wrote", "Doctor Zhivago", check_synonyms=False, head_feature="person",
                          tail_feature="book")
    graph.add_new_triplet("Vladimir Mayakovsky", "wrote", "Cloud in Pants", check_synonyms=False, head_feature="person",
                          tail_feature="book")
    graph.add_new_triplet("Doctor Zhivago", "has_character", "Yuri Zhivago", check_synonyms=False, head_feature="book",
                          tail_feature="character")
    graph.add_new_triplet("Cloud in Pants", "has_character", "Vladimir", check_synonyms=False, head_feature="book",
                          tail_feature="character")
    graph.add_new_triplet("Yuri Zhivago", "is_a", "character", check_synonyms=False, head_feature="character",
                          tail_feature="type")
    graph.add_new_triplet("Vladimir", "is_a", "character", check_synonyms=False, head_feature="character",
                          tail_feature="type")
    graph.add_new_triplet("Boris Pasternak", "inspired_by", "Leo Tolstoy", check_synonyms=False, head_feature="person",
                          tail_feature="person")
    graph.add_new_triplet("Vladimir Mayakovsky", "admired", "Anton Chekhov", check_synonyms=False,
                          head_feature="person", tail_feature="person")
    graph.add_new_triplet("Doctor Zhivago", "referenced_by", "War and Peace", check_synonyms=False, head_feature="book",
                          tail_feature="book")
    graph.add_new_triplet("Cloud in Pants", "referenced_by", "Eugene Onegin", check_synonyms=False, head_feature="book",
                          tail_feature="book")

    graph.add_new_triplet("Sergey Dovlatov", "is_a", "author", check_synonyms=False, head_feature="person",
                          tail_feature="type")
    graph.add_new_triplet("Lyudmila Ulitskaya", "is_a", "author", check_synonyms=False, head_feature="person",
                          tail_feature="type")
    graph.add_new_triplet("Daniil Kharms", "is_a", "author", check_synonyms=False, head_feature="person",
                          tail_feature="type")
    graph.add_new_triplet("Ivan Bunin", "is_a", "author", check_synonyms=False, head_feature="person",
                          tail_feature="type")
    graph.add_new_triplet("Varlam Shalamov", "is_a", "author", check_synonyms=False, head_feature="person",
                          tail_feature="type")
    graph.add_new_triplet("The Suitcase", "is_a", "novel", check_synonyms=False, head_feature="book",
                          tail_feature="type")
    graph.add_new_triplet("Sonechka", "is_a", "novel", check_synonyms=False, head_feature="book", tail_feature="type")
    graph.add_new_triplet("Today I Wrote Nothing", "is_a", "short_story", check_synonyms=False, head_feature="book",
                          tail_feature="type")
    graph.add_new_triplet("Dark Avenues", "is_a", "novel", check_synonyms=False, head_feature="book",
                          tail_feature="type")
    graph.add_new_triplet("Kolyma Tales", "is_a", "novel", check_synonyms=False, head_feature="book",
                          tail_feature="type")
    graph.add_new_triplet("Sergey Dovlatov", "wrote", "The Suitcase", check_synonyms=False, head_feature="person",
                          tail_feature="book")
    graph.add_new_triplet("Lyudmila Ulitskaya", "wrote", "Sonechka", check_synonyms=False, head_feature="person",
                          tail_feature="book")
    graph.add_new_triplet("Daniil Kharms", "wrote", "Today I Wrote Nothing", check_synonyms=False,
                          head_feature="person", tail_feature="book")
    graph.add_new_triplet("Ivan Bunin", "wrote", "Dark Avenues", check_synonyms=False, head_feature="person",
                          tail_feature="book")
    graph.add_new_triplet("Varlam Shalamov", "wrote", "Kolyma Tales", check_synonyms=False, head_feature="person",
                          tail_feature="book")
    graph.add_new_triplet("Sonechka", "has_character", "Sonia", check_synonyms=False, head_feature="book",
                          tail_feature="character")
    graph.add_new_triplet("Kolyma Tales", "has_character", "Prisoner", check_synonyms=False, head_feature="book",
                          tail_feature="character")
    graph.add_new_triplet("Dark Avenues", "has_character", "Nikolai", check_synonyms=False, head_feature="book",
                          tail_feature="character")
    graph.add_new_triplet("Sonia", "is_a", "character", check_synonyms=False, head_feature="character",
                          tail_feature="type")
    graph.add_new_triplet("Prisoner", "is_a", "character", check_synonyms=False, head_feature="character",
                          tail_feature="type")
    graph.add_new_triplet("Nikolai", "is_a", "character", check_synonyms=False, head_feature="character",
                          tail_feature="type")
    graph.add_new_triplet("Sergey Dovlatov", "admired", "Vladimir Nabokov", check_synonyms=False, head_feature="person",
                          tail_feature="person")
    graph.add_new_triplet("Lyudmila Ulitskaya", "inspired_by", "Leo Tolstoy", check_synonyms=False,
                          head_feature="person", tail_feature="person")
    graph.add_new_triplet("Daniil Kharms", "admired", "Anton Chekhov", check_synonyms=False, head_feature="person",
                          tail_feature="person")
    graph.add_new_triplet("Ivan Bunin", "inspired_by", "Ivan Turgenev", check_synonyms=False, head_feature="person",
                          tail_feature="person")
    graph.add_new_triplet("Varlam Shalamov", "admired", "Fyodor Dostoevsky", check_synonyms=False,
                          head_feature="person", tail_feature="person")
    graph.add_new_triplet("The Suitcase", "referenced_by", "Dead Souls", check_synonyms=False, head_feature="book",
                          tail_feature="book")
    graph.add_new_triplet("Sonechka", "referenced_by", "Anna Karenina", check_synonyms=False, head_feature="book",
                          tail_feature="book")
    graph.add_new_triplet("Today I Wrote Nothing", "referenced_by", "The Seagull", check_synonyms=False,
                          head_feature="book", tail_feature="book")
    graph.add_new_triplet("Dark Avenues", "referenced_by", "Lolita", check_synonyms=False, head_feature="book",
                          tail_feature="book")
    graph.add_new_triplet("Kolyma Tales", "referenced_by", "Crime and Punishment", check_synonyms=False,
                          head_feature="book", tail_feature="book")
    graph.add_new_triplet("Sonia", "inspired_by", "Anna Karenina", check_synonyms=False, head_feature="character",
                          tail_feature="character")
    graph.add_new_triplet("Prisoner", "inspired_by", "Pierre Bezukhov", check_synonyms=False, head_feature="character",
                          tail_feature="character")
    graph.add_new_triplet("Nikolai", "admired", "Vladimir", check_synonyms=False, head_feature="character",
                          tail_feature="character")

    graph.add_loop_reversed_triplet()
    #graph.print()

    return graph
