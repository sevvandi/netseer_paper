import igraph as ig
import pickle


def read_graph_list(filenames: list[str]) -> list[ig.Graph]:
    """Reads a list graphs from disk.

    The graphs are read in index order, therefore the first graph in the time-series is at index 0.

    Args:
        filenames: A list of absolute paths to ig.read compatible graphs. E.g. *.gml files.

    Returns:
        A list of read graphs.
    """
    # read a list of saved graph files
    print(f"Reading in graph files: {filenames}")
    return [ig.read(filename) for filename in filenames]


def read_pickled_list(filename: str) -> list[ig.Graph]:
    """Reads a saved list of graphs from a .pkl file.

    Args:
        filename: Absolute path to the .pkl file.

    Returns:
        A list of graphs.
    """
    # read a graph list that is stored in a single pickled file
    print(f"Reading pickled graph list from {filename}")
    return pickle.load(filename)
