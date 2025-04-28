import requests
from bs4 import BeautifulSoup
from urllib.parse import quote
from kg_library import get_config

class WikidataExtractor:
    def __init__(self):
        self.endpoint = "https://query.wikidata.org/sparql"
        self.openlibrary_base = "https://openlibrary.org"
        self.headers = {
            "User-Agent": f"BookInfoBot/1.0 ({get_config()['email']})",
            "Accept": "application/json"
        }

    def get_book_info(self, title: str) -> dict[str, list[dict]]:
        query = self.build_query(title)
        results = self.execute_query(query)
        parsed = self.parse_results(results)
        if not parsed["characters"]:
            characters = self.get_openlibrary_characters(title)
            for char in characters:
                for author in parsed["authors"]:
                    if author["label"].lower().strip() not in char["label"].lower().strip():
                        parsed["characters"].append(char)
        return parsed


    def build_query(self, title: str) -> str:
        return f"""
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
          {{ 
            ?work wdt:P50 ?value.  # автор
            ?value wdt:P31 wd:Q5.  # должен быть человеком
            BIND(wd:P50 AS ?property) 
          }}
          UNION
          {{ 
            ?work wdt:P577 ?value.  # дата публикации
            BIND(wd:P577 AS ?property) 
          }}
          UNION
          {{ 
            ?work wdt:P495 ?value.  # страна происхождения
            BIND(wd:P495 AS ?property) 
          }}
          UNION
          {{ 
            ?work wdt:P407 ?value.  # язык
            BIND(wd:P407 AS ?property) 
          }}

          # Персонажи
          UNION
          {{ 
            ?work wdt:P1441|wdt:P674 ?value.
            ?value wdt:P31/wdt:P279* wd:Q95074.  # вымышленные персонажи
            BIND(wd:P1441 AS ?property)
          }}

          # Места действия
          UNION
          {{ 
            ?work wdt:P840 ?value.
            BIND(wd:P840 AS ?property) 
          }}

          # Метки
          SERVICE wikibase:label {{ 
            bd:serviceParam wikibase:language "en". 
            ?property rdfs:label ?propertyLabel.
            ?value rdfs:label ?valueLabel.
          }}

          # Фильтры для чистоты данных
          FILTER EXISTS {{ ?value rdfs:label ?valLabel. FILTER(LANG(?valLabel) = "en") }}
        }}
        ORDER BY ?property ?valueLabel
        """

    def execute_query(self, query: str) -> list[dict]:
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
            print(f"SPARQL query failed: {e}")
            return []

    def parse_results(self, results: list[dict]) -> dict[str, list[dict]]:
        book_info = {
            "authors": [],
            "publication_dates": [],
            "countries": [],
            "languages": [],
            "characters": [],
            "locations": []
        }

        for item in results:
            prop = item.get("property", {}).get("value", "")
            value = {
                "id": item.get("value", {}).get("value", ""),
                "label": item.get("valueLabel", {}).get("value", "")
            }

            if "P50" in prop:
                book_info["authors"].append(value)
            elif "P577" in prop:
                book_info["publication_dates"].append(value)
            elif "P495" in prop:
                book_info["countries"].append(value)
            elif "P407" in prop:
                book_info["languages"].append(value)
            elif "P1441" in prop or "P674" in prop:
                book_info["characters"].append(value)
            elif "P840" in prop:
                book_info["locations"].append(value)

        return book_info

    def get_openlibrary_characters(self, title: str) -> list[dict]:
        try:
            search_url = f"{self.openlibrary_base}/search.json?q={quote(title)}"
            search_res = requests.get(search_url, timeout=15).json()

            if not search_res.get('docs'):
                return []

            work_key = search_res['docs'][0]['key']
            work_url = f"{self.openlibrary_base}{work_key}.json"
            work_data = requests.get(work_url, timeout=15).json()

            characters = []

            if 'subject_people' in work_data:
                characters.extend([{
                    "id": f"ol:{p.split('/')[-1]}",
                    "label": p.replace(' (Person)', ''),
                    "source": "openlibrary"
                } for p in work_data['subject_people'] if isinstance(p, str)])

            html_url = f"{self.openlibrary_base}{work_key}"
            html = requests.get(html_url, timeout=15).text
            soup = BeautifulSoup(html, 'html.parser')

            for h2 in soup.find_all('h2'):
                if 'character' in h2.text.lower():
                    for li in h2.find_next_sibling('ul').find_all('li'):
                        char_name = li.text.strip()
                        characters.append({
                            "id": f"ol:char_{hash(char_name)}",
                            "label": char_name,
                            "source": "openlibrary_html"
                        })

            return characters

        except Exception as e:
            print(f"OpenLibrary query failed: {e}")
            return []

    def print(self, title : str) -> None:
        info = api.get_book_info(title)

        print("\nРезультаты:")
        print(f"Авторы: {', '.join(a['label'] for a in info['authors'])}")
        print(f"Даты публикации: {', '.join(d['label'] for d in info['publication_dates'])}")
        print(f"Страны: {', '.join(c['label'] for c in info['countries'])}")
        print(f"Языки: {', '.join(l['label'] for l in info['languages'])}")

        print("\nПерсонажи:")
        for char in info["characters"]:
            print(f"- {char['label']} ({char['id']})")

        print("\nМеста действия:")
        for loc in info["locations"]:
            print(f"- {loc['label']} ({loc['id']})")

if __name__ == "__main__":
    api = WikidataExtractor()
    api.print("1984")
