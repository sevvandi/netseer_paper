# Netseer repository

This is a repository containing the code for both `netseer` R and `netseer` Python.
This repository contains the:

- Source code for the R implementation under `./netseer-r/`
- Source code for the Python implementation under `./netseer-py/`
- The JOSS paper files under `./paper/`

The source code for the R and Python versions is a direct drag and drop from their respective repositories as of 2025/10/03.

## Installation

### From a package repository

Both packages are available on CRAN and PYPI under `netseer`

```bash
install.packages("netseer") # R
pip install netseer # Python
```

### Building from source

Download the repository:

``` bash
git clone git@github.com:sevvandi/netseer_paper.git
cd netseer_paper
```

Inside of the netseer_paper project root, either Python or R libraries can be built.
Alternatively for R, the package can be built directly from GitHub using devtools (See R section below).

#### Python

The Python project assumes you are using uv.
Installing directly from GitHub:

```bash
uv pip install "git+https://github.com:sevvandi/netseer_paper/netseer-py"
```

For building netseer locally using uv:

``` bash
cd netseer-py
uv sync
uv build
```

This will generate a wheel and tar inside dist/.
In another UV project, you can then:

``` bash
uv add --editable path/to/netseer_paper/netseer-py
```

This will install netseer from the `./netseer_paper/netseer-py`.
The `path/to/netseer_paper` can be relative e.g. `../netseer_paper`.

#### R

For R, the easiest method is building directly from GitHub using the remotes package:

``` R
install.packages("remotes")
remotes::install_github("sevvandi/netseer-r/netseer_paper")
```

Afterwards, `netseer` should be accessible.

For Windows, this will require [RTools](https://cran.r-project.org/bin/windows/Rtools/) as some of the R code is using C.  
Linux installs may need a C compiler.

For building netseer locally using R:

``` R
remotes::install_local(
  path = "/path/to/netseer_paper/netseer-r",
  dependencies = TRUE
)
```
