import pandas as pd
from datasets import load_dataset
from kg_library import get_config
from kg_library.utils import PathManager

literature_dataset = load_dataset(
    "kingkangkr/book_summary_dataset",
    cache_dir=PathManager.get_datasets_cache_path(),
)

data_frame = pd.DataFrame(literature_dataset['train'])

def main():
    print("Первые 5 записей:")
    pd.set_option('display.max_columns', None)
    print(data_frame.head())
    print("\ndata frame info:")
    print(data_frame.info())
    print("\ndata frame describe:")
    print(data_frame.describe())
    print("\ngenres:")
    print(data_frame['Parsed Genres'].unique())

if __name__ == "__main__":
    main()