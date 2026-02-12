**TODO:** after redoing netseer-r/README.md, redo netseer-py/README.md in the same fashion

# Netseer

_Netseer_ is a software package for predicting new graphs from a given time series of graphs.

The underlying prediction algorithm combines time series modelling
with an adapted form of Flux Balance Analysis,
an approach widely used in biochemistry for reconstructing metabolic networks from partial information.
A comprehensive description of the algorithm is given in:

* Predicting Graph Structure via Adapted Flux Balance Analysis.  
  Lecture Notes in Computer Science (LNCS), Vol.&nbsp;16370, 2026.  
  DOI: [10.1007/978-981-95-4969-6_27](https://doi.org/10.1007/978-981-95-4969-6_27); arXiv: [2507.05806](https://arxiv.org/abs/2507.05806)

## Installation - TODO

TODO: need matplotlib as a dependency  
TODO: need natsort    as a dependency  
TODO: need pathlib    as a dependency  

This package is available for installation on PyPI:

``` Bash
pip install netseer
```

Alternatively, `netseer` can be built from source from GitHub:  

``` Bash
pip install "git+https://github.com/sevvandi/netseer_paper.git#subdirectory=netseer-py"
```

## Available Functions - TODO

| Function                  | Summary                                                         |
| ---                       | ---                                                             |
| `read_graph_list()`       | load user provded graphs                                        |
| `predict_graph()`         | predict the next graph in a sequence                            |
| `measure_error()`         | return the vertex error and edge error between two graphs       |
| `generate_graph_linear()` | generate a time series of random graphs that grow linearly      |
| `generate_graph_exp()`    | generate a time series of random graphs that grow exponentially |

Documentation for the above functions is available in [netseer.pdf](TODO:netseer.pdf) (TODO: update link documentation PDF)

## Example

Goal:

* Load 20 graphs from the file system.
* Use graphs 1 to 19 to predict the 20th graph.
* Compare the actual 20th graph to the newly predicted 20th graph.

Before starting, download the [data.zip](./data.zip) directory under `/netseer-paper/netseer-py/`. This directory contains 20 example graphs. Extract the zip to your project root.  

TODO: rename `data.zip` to `example_graphs.zip`

``` Python
import netseer as ns
import natsort # pip install natsort


# Create a path to the extracted data directory. Replace "data" with the relative path to data 
directory_path = Path.cwd() / Path("data")
# Use glob to get a list of absolute paths for only .gml files.
graph_files = list(directory_path.glob("*.gml"))
# Optionally you may need to sort the graph names.
graph_files = natsort.natsorted(graph_files)
graph_list = ns.read_graph_list(graph_files)

# TODO: ensure the same h setting is used as in the R version
# Predict the 20th graph using graphs 1 to 19.   
predicted_graph = ns.predict_graph(graph_list[0:19], h=1, weights_option = 1)

# Compare the 20th actual graph and the predicted 20th graph by checking the vertex and edge error.
vertex_error, edge_error = ns.measure_error(graph_list[19], predicted_graph)
print(f"Vertex Error: {vertex_error} |  Edge Error: {edge_error}")
```


TODO:
```
>>> directory_path = Path.cwd() / Path("example_graphs")
Traceback (most recent call last):
  File "<python-input-2>", line 1, in <module>
    directory_path = Path.cwd() / Path("example_graphs")
                     ^^^^
NameError: name 'Path' is not defined
```

TODO: change to `directory_path = pathlib.Path.cwd() / pathlib.Path("example_graphs")`

