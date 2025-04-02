from datasets import load_dataset
import pandas as pd

literature_dataset = load_dataset(
    "kingkangkr/book_summary_dataset",
    split="train[:1%]",
    cache_dir="/cache/datasets"
)

data_frame = pd.DataFrame(literature_dataset)

def main():
    print("Первые 5 записей:")
    pd.set_option('display.max_columns', None)
    print(data_frame.head())
    print("\nИнформация о датафрейме:")
    print(data_frame.info())
    print("\nОсновные статистики:")
    print(data_frame.describe())
    print("\nУникальные жанры:")
    print(data_frame['Parsed Genres'].unique())
    fantasy_books = data_frame[data_frame['Parsed Genres'] == 'Fantasy']
    print("\nКоличество фэнтези-книг:", len(fantasy_books))
    genre_counts = data_frame.groupby('Parsed Genres').size()
    print("\nКоличество книг по жанрам:")
    print(genre_counts)

if __name__ == "__main__":
    main()