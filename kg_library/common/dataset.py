from datasets import load_dataset
import pandas as pd

literature_dataset = load_dataset("kingkangkr/book_summary_dataset")

data_frame = pd.DataFrame(literature_dataset['train'])

def main():
    print("Первые 5 записей:")
    pd.set_option('display.max_columns', None)  # Отключаем ограничение на кол-во столбцов
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
    data_frame['summary_length'] = data_frame['Plot summary'].apply(lambda x: len(x.split()))
    print("\nСредняя длина аннотации:", data_frame['summary_length'].mean())
    top_longest = data_frame.sort_values('summary_length', ascending=False).head(3)
    print("\nТоп-3 самых длинных аннотации:")
    print(top_longest[['title', 'summary_length']])

if __name__ == "__main__":
    main()