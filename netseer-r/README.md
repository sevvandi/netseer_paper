# Netseer

*Netseer* is an open-source software package for R that models a temporal sequence of graphs and predicts new graphs at future time steps.
The underlying algorithm is comprised of time series modelling combined with an adapted form of Flux Balance Analysis,
an approach used in biochemistry for reconstructing metabolic networks from partial information.
A comprehensive description of the algorithm is given in:

- Predicting Graph Structure via Adapted Flux Balance Analysis.  
  Lecture Notes in Computer Science (LNCS), Vol. 16370, 2026.  
  DOI: [10.1007/978-981-95-4969-6_27](https://doi.org/10.1007/978-981-95-4969-6_27);
  arXiv: [2507.05806](https://arxiv.org/abs/2507.05806)

## Installation

The `netseer` package can be installed via CRAN or via GitHub.
The latter will have the most up to date version.

- Installation via CRAN:

``` r
install.packages("netseer")
```

- Installation via GitHub:

``` r
install.packages("remotes")
library("remotes")
remotes::install_github("sevvandi/netseer_paper/netseer-r")
```

When building from GitHub, a C++ compiler and the `curl` library are required to build the dependencies.

* Windows: install [RTools](https://cran.r-project.org/bin/windows/Rtools/rtools44/rtools.html)
* MacOS: install Xcode; command line example: `xcode-select --install`
* Ubuntu/Debian: `sudo apt install build-essential libcurl4-openssl-dev` (**TODO:** check)
* Fedora/RHEL/CentOS: `sudo dnf install gcc-c++ libstdc++-devel libcurl-devel`

## Available Functions

* `read_graph_list()`       - load user provided graphs in alphanumeric order
* `predict_graph()`         - predict the next graph in a sequence.
* `measure_error()`         - return the vertex error and edge error between two graphs
* `generate_graph_linear()` - generate a time series of random graphs that grow linearly
* `generate_graph_exp()`    - generate a time series of random graphs that grow exponentially

Documentation for the above functions is available in the [Documentation PDF](./docs/netseer.pdf).

**TODO:** update documentation to make sure all the functions are listed and described

**TODO:** update documentation to bump the version number (eg. 0.2.1)


## Example

Steps:

- load 20 graphs from the file system
- use graphs 1 to 19 to predict graph 20
- compare predicted graph 20 to actual graph 20

Before starting, download the [example_graphs.zip](./example_graphs.zip) and extract the zip to your project root.
The zip contains 20 example graphs.

------------------------------------------------------------------------

``` r
library("netseer")

## load 20 graphs from the example_graphs directory;
## replace ./example_graphs/ in file.path() with a path to the example_graphs directory

path_to_graphs <- file.path("./example_graphs/")
graph_list <- netseer::read_graph_list(path_to_graphs = path_to_graphs, format = "gml")

## use graphs 1 to 19 to predict graph 20;
## h=1 means predict 1 step into the future;
## weights_opt=7 sets the edge weight of the last seen graph as 1, otherwise 0

predicted_graph <- netseer::predict_graph(graph_list[1:19], weights_opt=7, h=1)

## compare predicted graph 20 to actual graph 20 by checking the vertex and edge errors

output <- netseer::measure_error(graph_list[[20]], predicted_graph[[1]])
print(output$vertex_err)
print(output$edge_err)
```



