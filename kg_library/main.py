from db import Neo4jConnection
from models.entity_extraction.rebel import from_small_text_to_kb, raw_documents

# метод-заглушка
def extract_relations(text):
    return [
        ("George Orwell", "wrote", "1984"),
        ("1984", "belongs_to", "dystopia")
    ]

def extract_relations_with_rebel(text):
    result_tuple = []
    for doc in text:
        kb = from_small_text_to_kb(doc.page_content, verbose=True)
        print(kb)
        result_tuple += [tuple(r.values()) for r in kb.relations]
    return result_tuple

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
    #relations = extract_relations(text)
    relations = extract_relations_with_rebel(raw_documents)
    conn = Neo4jConnection()
    insert_relations_to_neo4j(relations, conn)
    print("Связи успешно вставлены в граф!")
    conn.close()

if __name__ == "__main__":
    main()
