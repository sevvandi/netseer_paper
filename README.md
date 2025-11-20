# Graph Prediction Using Netseer

This repository contains the code for both the R and Python implementations of the `netseer` package.

- The `netseer-r` folder contains the source code for the R implementation
- The `netseer-py` foldr contains the source code for the Python implementation
- The `paper` folder contains the JOSS paper sources


# Installation - R

Netseer can be installed directly from CRAN or via building from source.

In either case, first ensure that a C++ compiler is available on your system and relevant R development files are also installed.
If you are using Windows, install the [RTools](https://cran.r-project.org/bin/windows/Rtools/) toolchain.
TODO: If you are using Ubuntu or Debian ...
TODO: If you are using Fedora or Red Hat ...


## Install from CRAN

To install from CRAN, use the following command in the R shell:

``` R
install.packages("netseer")
```

## Install from source

Netseer can be built from source using two ways.
The first way is to build directly from GitHub using the _remotes_ package within the R shell:

``` R
install.packages("remotes")
remotes::install_github("sevvandi/netseer_paper/netseer-r")
```

**TODO**: is the second local method really necessary?

Alternatively, the package can be built locally as follows.
First, clone the `netseer_paper` repository on the command line:

```bash
mkdir ~/netseer
cd ~/netseer
git clone https://github.com/sevvandi/netseer_paper.git
```

Then use the following command in the R shell:

```R
remotes::install_local(path = "~/netseer/netseer_paper/netseer-r", dependencies = TRUE)
```

&nbsp;



# Installation - Python

Netseer can be installed directly from PyPI or via building from source.

To install from PyPI:

```bash
pip install netseer
```

There are two ways to build from source.
First, ensure that the _uv_ tool is installed.

The package can be built directly from GitHub via the following command:

```bash
uv pip install "git+https://github.com:sevvandi/netseer_paper/netseer-py"
```

**TODO**: is the second local method really necessary?

Alternatively, the package can be built locally as follows.

```bash
mkdir ~/netseer
cd ~/netseer
git clone https://github.com/sevvandi/netseer_paper.git
cd netseer_paper
cd netseer-py
uv sync
uv build
```

This will generate a wheel and tar inside dist/.
In another UV project, you can then:
(TODO: check and fix this command)

``` bash
uv add --editable netseer_paper/netseer-py
```

This will install netseer from the `./netseer_paper/netseer-py`.
The `path/to/netseer_paper` can be relative e.g. `../netseer_paper`.

