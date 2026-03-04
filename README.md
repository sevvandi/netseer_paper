# Graph Prediction Using Netseer

* [`netseer-r`](./netseer-r/) contains the R implementation
* [`paper`](./paper/) contains JOSS paper sources

---

*Netseer* is an open-source software package for R that models a temporal sequence of graphs and predicts new graphs at future time steps.
The underlying algorithm is comprised of time series modelling combined with an adapted form of Flux Balance Analysis,
an approach used in biochemistry for reconstructing metabolic networks from partial information.
A comprehensive description of the algorithm is given in:

* Predicting Graph Structure via Adapted Flux Balance Analysis.  
  Lecture Notes in Computer Science (LNCS), Vol. 16370, 2026.  
  DOI: [10.1007/978-981-95-4969-6_27](https://doi.org/10.1007/978-981-95-4969-6_27);
  arXiv: [2507.05806](https://arxiv.org/abs/2507.05806)

## Installation

The `netseer` package can be installed via CRAN or via GitHub.
The latter will have the most up to date version.

* Installation via CRAN:

``` r
install.packages("netseer")
```

* Installation via GitHub:

``` r
install.packages("remotes")
library("remotes")
remotes::install_github("sevvandi/netseer_paper/netseer-r", Ncpus=3)
```

**Caveat**: When building from GitHub, a C++ compiler and the `curl` library may be needed to build the dependencies.

* Windows: install [RTools](https://cran.r-project.org/bin/windows/Rtools/rtools44/rtools.html)
* MacOS: install Xcode; command line example: `xcode-select --install`
* Ubuntu/Debian: `sudo apt install build-essential libcurl4-openssl-dev`
* Fedora/RHEL/CentOS: `sudo dnf install gcc-c++ libstdc++-devel libcurl-devel`

## Available Functions

* `read_graph_list()`       - load user provided graphs in alphanumeric order **TODO:** check
* `predict_graph()`         - predict the next graph in a sequence.
* `measure_error()`         - return the vertex error and edge error between two graphs.
* `generate_graph_list()`   - generate a time series of random graphs that grow either linearly or exponentially.
* `save_graphs()`           - save a graph or list of graphs to the file system in a specified format

**TODO:** check R package documentation to make sure all the functions are listed and described

The above functions are described in the [R package documentation](./netseer-r/docs/netseer.pdf).


## Example

Steps:

* load 20 graphs from the file system
* use graphs 1 to 19 to predict graph 20
* compare predicted graph 20 to actual graph 20

Before starting, download the [example_graphs.zip](./example_graphs.zip) and extract the zip to your project root.
The zip contains 20 example graphs.

``` r
library("netseer")

## Create an absolute path to the example_graphs directory.
## Replace "./example_graphs" with the relative path to the example_graphs directory.

path_to_graphs <- normalizePath("./example_graphs")

## Load the example graphs into R.

graph_list <- netseer::load_graphs(use_directory = path_to_graphs, format = "gml")

## Predict the 20th graph using graphs 1 to 19.
## h=1 means predict 1 step into the future. So predict the 20th graph.
## weights_opt=7 sets the edge weight of the last seen graph as 1, otherwise 0.

predicted_graph <- netseer::predict_graph(graph_list[1:19], weights_opt=7, h=1)

output <- netseer::measure_error(graph_list[[20]], predicted_graph[[1]])
print(output$vertex_err)   # possible output: 0.053
print(output$edge_err)     # possible output: 0.013

# To save a graph or graph list back into gml.
# Create a path to a directory tosave the graphs to. Add the name of the graph to the end.
saved_path <- normalizePath("./saved/graph")
netseer::save_graphs(predicted_graph, saved_path, ".gml", "gml")

```

Alternatively to loading the graphs from a file, synthetic graphs can be generated from within netseer.  
These graphs are used in the same flow as the graphs loaded from the file system and can be saved using the same `save_graphs()`.  

``` r
# Generate 20 graphs.
graph_list_exp <- netseer::generate_graph_list(num_graphs = 20, mode = "exp")

predicted_graph <- netseer::predict_graph(graph_list_exp[1:19], weights_opt=7, h=1)

```
