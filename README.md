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
remotes::install_github("sevvandi/netseer_paper/netseer-r", Ncpus=4)
```

* **Caveat**:
When building from GitHub, a C++ compiler and/or the _curl_ library may need to be installed first in order to build the dependencies;
these can be installed via:
  * Windows: install [RTools](https://cran.r-project.org/bin/windows/Rtools/rtools44/rtools.html)
  * MacOS: install Xcode; command line example: `xcode-select --install`
  * Ubuntu/Debian: `sudo apt install build-essential libcurl4-openssl-dev`
  * Fedora/RHEL/CentOS: `sudo dnf install gcc-c++ libstdc++-devel libcurl-devel`


## Available Functions

Summary of available functions:

* `load_graphs()`         - load graphs from the file system
* `save_graphs()`         - save a graph or list of graphs to the file system
* `predict_graph()`       - predict the next graph in a sequence
* `measure_error()`       - return the vertex error and edge error between two graphs
* `generate_graph_list()` - generate a time series of random graphs that grow either linearly or exponentially

The above functions are described in the [R package documentation](./netseer-r/docs/netseer.pdf).


## Example

The example below loads 20 graphs from the file system and uses graphs 1 to 19 to predict graph 20.
The predicted graph 20 is compared to the actual graph 20 via measuring vertex and edge errors.

Before starting, download [example_graphs.zip](./netseer-r/example_graphs.zip) which contains the 20 example graphs.  
Extract the graphs to your project root.

``` r
library("netseer")

# create an absolute path to the example_graphs directory; change "./example_graphs" to reflect your setup
path_to_graphs <- normalizePath("./example_graphs")

# load the example graphs
graph_list <- netseer::load_graphs(use_directory = path_to_graphs, format = "gml")

# use graphs 1 to 19 to predict graph 20;
# h=1 means predict 1 step into the future;
# weights_opt=7 sets the edge weight of the last seen graph as 1, otherwise 0
predicted_graph <- netseer::predict_graph(graph_list[1:19], weights_opt=7, h=1)

# compare predicted graph 20 to actual graph 20 by checking the vertex and edge errors
output <- netseer::measure_error(graph_list[[20]], predicted_graph[[1]])

print(output$vertex_err)   # possible output: 0.053
print(output$edge_err)     # possible output: 0.013

# save the predicted graph as a file
saved_path <- normalizePath("./predicted_graph")
netseer::save_graphs(predicted_graph, saved_path, ".gml", "gml")

```

It is also possible to generate a list of random graphs that grow in linear or exponential manner.
For example:

``` r

# generate 20 random graphs that grow in an exponential manner
graph_list <- netseer::generate_graph_list(num_graphs = 20, mode = "exp")

```
