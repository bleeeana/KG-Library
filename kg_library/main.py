from kg_library import AppFacade

def main():
    app_facade = AppFacade()
    print("Generating graph...")
    app_facade.generate_graph_for_learning()

if __name__ == "__main__":
    main()
