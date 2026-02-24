import igraph as ig
import pickle
import natsort
from pathlib import Path


def read_graph_list(
    filepath: str = "", sort: bool = True, graph_type: str = "*.gml"
) -> list[ig.Graph]:
    """Reads a list graphs from disk.

    The graphs are read in index order, therefore the first graph in the time-series is at index 0.

    Args:
        filepath: The relative path from the root directory to the graphs directory as a string. E.g. "graph_files"
        sort: A boolean for sorting the graphs contained in the graphs directory alphanumerically. 1 to sort. 0 for OFF.
        graph_type: A string using wildcards to denote the file type of the graphs to be loaded. Must be iGraph compatible e.g. "*.gml"
            The "*.gml" means all gml files found in the filepath are loaded.

    Returns:
        A list of read graphs.
    """
    filenames = list()
    directory_path = Path.cwd() / Path(filepath)
    # Use glob to get a list of absolute paths for only .gml files.
    filenames = list(directory_path.glob(graph_type))

    if not filenames:
        print(
            f"Error:Either directory doesn't exist, or no files found in directory in direction {directory_path}."
        )
        quit()

    # Optionally you may need to sort the graph names.
    if sort:
        filenames = natsort.natsorted(filenames)

    # read a list of saved graph files
    print(f"Reading in graph files: {filenames}")
    return [ig.read(filename) for filename in filenames]


def read_pickled_list(filepath: str) -> list[ig.Graph]:
    """Reads a saved list of graphs from a .pkl file.

    Args:
        filepath: The relative path from the root directory to the pickle file. e.g. "pickled_graphs"
    Returns:
        A list of graphs.
    """
    filename = Path.cwd() / Path(filepath)

    # read a graph list that is stored in a single pickled file
    print(f"Reading pickled graph list from {filename}")
    try:
        with open(filename, "rb") as file:
            data = pickle.load(file)
            return data
    except FileNotFoundError:
        print(f"Error: {filename} not found.")
        quit()
    except Exception as e:
        print(f" An error occurred. {e}")
        quit()
