from db import Neo4jConnection


def main():
    conn = Neo4jConnection()
    conn.close()

if __name__ == "__main__":
    main()
