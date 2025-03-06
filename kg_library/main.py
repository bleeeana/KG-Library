from db import Neo4jConnection

def extract_relations(text):
    return [
        ("George Orwell", "wrote", "1984"),
        ("1984", "belongs_to", "dystopia")
    ]


def insert_relations_to_neo4j(relations, conn):
    for subj, rel, obj in relations:
        query = (
            "MERGE (a:Entity {name: $subj}) "
            "MERGE (b:Entity {name: $obj}) "
            "MERGE (a)-[r:RELATION {type: $rel}]->(b)"
        )
        conn.run_query(query, {"subj": subj, "rel": rel, "obj": obj})


def main():
    text = "George Orwell wrote 1984, a dystopian novel."
    relations = extract_relations(text)
    conn = Neo4jConnection()
    insert_relations_to_neo4j(relations, conn)
    print("Связи успешно вставлены в граф!")
    conn.close()

if __name__ == "__main__":
    main()
