from kg_library import AppFacade, Neo4jConnection
import argparse, os, zipfile, tempfile
from kg_library.common import GraphData

from kg_library.utils import VideoProcessor


class KnowledgeGraphGeneratorWrapper:
    def __init__(self, model_path: str = None):
        self.app_facade = AppFacade()
        if model_path and os.path.exists(model_path):
            if model_path.endswith(".zip"):
                self.load_model_from_zip(model_path)
            else:
                self.app_facade.load_model(model_path)
                print(f"Model loaded from {model_path}")

    def load_model_from_zip(self, model_path: str):
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                with zipfile.ZipFile(model_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)
                model_path = os.path.join(temp_dir, os.listdir(temp_dir)[0])
                if os.path.exists(model_path):
                    self.app_facade.load_model(model_path)
                    print(f"Model loaded from {model_path}")
                else:
                    print(f"Model not found in {model_path}")
        except Exception as e:
            print(f"Failed to load model from {model_path}: {e}")

    def save_model(self, output_path: str):
        if not self.app_facade.model or not self.app_facade:
            print("No model loaded")
            return
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with tempfile.TemporaryDirectory() as temp_dir:
            self.app_facade.save_model(os.path.join(temp_dir, "model.pt"), os.path.join(temp_dir, "graph.json"))
            with zipfile.ZipFile(output_path, 'w') as zip_ref:
                for root, dirs, files in os.walk(temp_dir):
                    for file in files:
                        zip_ref.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), temp_dir))

            print(f"Model saved to {output_path}")

    def process_text_file(self, file_path: str, confidence_threshold: float = 0.65, find_internal_links=False,
                          finetune=False, model_path="model_finetune.pt"):
        if not os.path.exists(file_path):
            print(f"File {file_path} not found")
            return None

        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()

        return self.app_facade.generate_graph_from_text(text, confidence_threshold, find_internal_links,
                                                        finetune=finetune, model_path=model_path)

    def process_audio_file(self, file_path: str, confidence_threshold: float = 0.65, find_internal_links=False,
                           finetune=False, model_path="model_finetune.pt"):
        if not os.path.exists(file_path):
            print(f"File {file_path} not found")
            return None

        return self.app_facade.generate_graph_from_audio(file_path, confidence_threshold,
                                                         link_prediction=find_internal_links, finetune=finetune,
                                                         model_path=model_path)

    def process_video_file(self, file_path: str, confidence_threshold: float = 0.65, link_prediction: bool = False,
                           finetune=False, model_path="model_finetune.pt"):
        if not os.path.exists(file_path):
            print(f"File {file_path} not found")
            return None

        return self.app_facade.generate_graph_from_text(VideoProcessor.extract_text_from_audio(video_path=file_path,
                                                                                               audio_processor=self.app_facade.audio_processor),
                                                        confidence_threshold, find_internal_links=link_prediction,
                                                        finetune=finetune, model_path=model_path)

    def learn_model(self, load_model_from_file=False, model_path="model.pt", load_triplets_from_file=False, load_graph=False,
                    graph_path="base_graph.json", finetune=False, dataset_size=200):
        self.app_facade.generate_graph_for_learning(load_graph=load_graph, graph_path=graph_path,
                                                    load_model_from_file=load_model_from_file,
                                                    load_triplets_from_file=load_triplets_from_file, finetune=finetune,
                                                    dataset_size=dataset_size, model_path=model_path)

    @staticmethod
    def save_graph_to_db(graph: GraphData):
        connection = Neo4jConnection()
        graph.fill_database(connection)


def main():
    parser = argparse.ArgumentParser(description="Literature Knowledge Graph Generator")

    parser.add_argument("--input", type=str, help="Path to input file (text, mp3, mp4)")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to model (.pt or .zip file)")
    parser.add_argument("--output", type=str, default="models/kg_model_updated.zip",
                        help="Path to save updated model")
    parser.add_argument("--confidence", type=float, default=0.65,
                        help="Confidence threshold for filtering triplets (0-1)")
    parser.add_argument("--link-prediction", type=bool, default=False,
                        help="Allow prediction of connections within the graph")
    parser.add_argument("--graph-path", type=str, default=None,
                        help="Path to existing graph to use for training (.json). If specified, the existing graph will be loaded.")
    parser.add_argument("--load-triplets", action="store_true",
                        help="Load saved triplets from JSON for training or finetuning")
    parser.add_argument("--learn", action="store_true",
                        help="Train model from scratch with base dataset")
    parser.add_argument("--finetune", action="store_true",
                        help="Finetune model on new data")
    parser.add_argument("--no-save", action="store_true",
                        help="Don't save updated model")
    parser.add_argument("--size-dataset", type=int, default=200, help="Size of learning dataset")
    parser.add_argument("--no-neo4j", action="store_true",
                        help="Don't save graph to Neo4j")
    args = parser.parse_args()

    if args.learn:
        print(f"Starting model training from scratch... {args.size_dataset} works will be used")
        processor = KnowledgeGraphGeneratorWrapper()

        load_model = False if args.model is None else True
        load_triplets = args.load_triplets
        load_graph = args.graph_path is not None
        graph_path = "base_graph.json" if args.graph_path is None else args.graph_path

        processor.learn_model(load_model_from_file=load_model, load_triplets_from_file=load_triplets,
                              load_graph=load_graph, graph_path=graph_path, finetune=args.finetune, dataset_size=args.size_dataset, model_path=args.model)
        if not args.no_save:
            print(f"Saving trained model to {args.output}...")
            processor.save_model(args.output)

        if not args.no_neo4j:
            print("Saving graph to Neo4j...")
            processor.save_graph_to_db(processor.app_facade.graph)

        print("Training completed.")

        return

    if args.input is None:
        print("Error: input file not specified (--input)")
        return

    if not os.path.exists(args.input):
        print(f"Error: file {args.input} not found")
        return

    file_extension = os.path.splitext(args.input)[1].lower()
    processor = KnowledgeGraphGeneratorWrapper(args.model)

    if file_extension in ['.txt', '.md', '.rst', '.tex']:
        print(f"Processing text file: {args.input}")
        graph = processor.process_text_file(file_path=args.input, confidence_threshold=args.confidence,
                                            find_internal_links=args.link_prediction, finetune=args.finetune,
                                            model_path=args.model)
    elif file_extension in ['.mp3', '.wav', '.ogg', '.flac']:
        print(f"Processing audio file: {args.input}")
        graph = processor.process_audio_file(file_path=args.input, confidence_threshold=args.confidence,
                                             find_internal_links=args.link_prediction, finetune=args.finetune,
                                             model_path=args.model)
    elif file_extension in ['.mp4', '.avi', '.mov', '.mkv']:
        print(f"Processing video file: {args.input}")
        graph = processor.process_video_file(file_path=args.input, confidence_threshold=args.confidence,
                                             link_prediction=args.link_prediction, finetune=args.finetune,
                                             model_path=args.model)
    else:
        print(f"Unsupported file format: {file_extension}")
        return

    if not args.no_neo4j:
        print("Saving graph to Neo4j...")
        processor.save_graph_to_db(graph)

    if not args.no_save:
        print(f"Saving updated model to {args.output}...")
        processor.save_model(args.output)

    print("Processing completed.")


if __name__ == "__main__":
    main()
