# Deku Neural Data Processing

## Description
We are working to design a program to process and analyze the electrophysiological signals being recorded in the Deku Lab at the University of Oregon. Given the use of experimental thin-film devices, having code that can be modified to understand better the data being recorded is of utmost importance.
## Getting Started
We are using Conda for package and environment management
### Setting Up the Conda Environment
To set up the project environment using Conda:

1. Ensure [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) is installed on your system.

2. Clone the repository:
```bash
git clone https://github.com/maxtenenbaum/deku_ephys.git
cd deku_ephys
```
3. Create the Conda environment from the environment.yml file:

```bash
conda env create -f environment.yml
```
4. Activate the environment:
```bash
conda activate deku_ephys_env
```
