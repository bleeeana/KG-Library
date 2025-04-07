import requests
from typing import Dict, List, Optional
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
from urllib.parse import quote  # Вместо requests.utils.quote

class WikidataLiteraryWork:
    def __init__(self):
        self.endpoint = "https://query.wikidata.org/sparql"
        self.headers = {
            "User-Agent": "LiteraryWorkBot/1.0 (example@example.com)",
            "Accept": "application/json"
        }

    def _execute_sparql(self, query: str) -> List[Dict]:
        try:
            response = requests.get(
                self.endpoint,
                headers=self.headers,
                params={"query": query, "format": "json"},
                timeout=30
            )
            response.raise_for_status()
            return response.json().get("results", {}).get("bindings", [])
        except Exception as e:
            print(f"Query failed: {e}")
            return []

    def get_work_info(self, title: str) -> List[Dict]:
        query = f"""
        SELECT DISTINCT ?property ?propertyLabel ?value ?valueLabel WHERE {{
          # Находим произведение по точному названию
          ?work rdfs:label "{title}"@en;
                wdt:P31/wdt:P279* ?type.

          # Все возможные типы литературных произведений
          VALUES ?literaryType {{
            wd:Q7725634    # literary work
            wd:Q47461344   # written work
            wd:Q8261       # book
            wd:Q1369       # book series
            wd:Q7725310    # creative work series
            wd:Q25379      # novel
            wd:Q49084      # short story
            wd:Q2831984    # novella
            wd:Q379885     # poem
            wd:Q5185279    # play
            wd:Q1983062    # allegory
          }}
          FILTER(?type = ?literaryType)

          # Основные свойства
          {{ ?work wdt:P50 ?value. BIND(wd:P50 AS ?property) }}
          UNION
          {{ ?work wdt:P577 ?value. BIND(wd:P577 AS ?property) }}
          UNION
          {{ ?work wdt:P495 ?value. BIND(wd:P495 AS ?property) }}
          UNION
          {{ ?work wdt:P407 ?value. BIND(wd:P407 AS ?property) }}
          UNION
          {{ ?work wdt:P840 ?value. BIND(wd:P840 AS ?property) }}

          # Метки
          SERVICE wikibase:label {{ 
            bd:serviceParam wikibase:language "en". 
            ?property rdfs:label ?propertyLabel.
            ?value rdfs:label ?valueLabel.
          }}
        }}
        ORDER BY ?property ?valueLabel
        LIMIT 200
        """
        return self._execute_sparql(query)

class OpenLibraryClient:
    def __init__(self):
        self.search_url = "https://openlibrary.org/search.json"
        self.base_url = "https://openlibrary.org"

    def get_work_subjects(self, title: str) -> Dict[str, List[str]]:
        try:
            response = requests.get(self.search_url, params={"title": title}, timeout=15)
            response.raise_for_status()
            docs = response.json().get("docs", [])
            if not docs:
                print("Книга не найдена.")
                return {}

            work_key = docs[0].get("key")
            if not work_key:
                print("Work key не найден.")
                return {}

            work_resp = requests.get(f"{self.base_url}{work_key}.json", timeout=15)
            work_resp.raise_for_status()
            data = work_resp.json()

            return {
                "title": [data.get("title")],
                "people": data.get("subject_people", []),
                "places": data.get("subject_places", []),
                "times": data.get("subject_times", []),
                "subjects": data.get("subjects", [])
            }

        except Exception as e:
            print(f"Ошибка при получении данных из Open Library: {e}")
            return {}


if __name__ == "__main__":
    client = OpenLibraryClient()

    info = client.get_work_subjects("A Clockwork Orange")

    if info:
        print(f"\n {info['title'][0]}")
        print("\n Люди (включая персонажей):")
        for person in info["people"]:
            print(f"- {person}")

        print("\nМеста:")
        for place in info["places"]:
            print(f"- {place}")

        print("\nЭпохи:")
        for time in info["times"]:
            print(f"- {time}")

        print("\nТемы / Категории:")
        for subject in info["subjects"]:
            print(f"- {subject}")
