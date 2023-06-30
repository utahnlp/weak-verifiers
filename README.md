# weak-verifiers
This repository contains the data and implementation for the ACL'23 Findings paper: "Verifying Annotation Agreement without Multiple Experts: A Case Study with Gujarati SNACS" 

## Setup 
Supported Python Version: 3.6+<br>
To get started with the project, follow the steps mentioned below:
1. Clone the repository to your local working directory.
  ```console
  foo@bar:~$ git clone https://github.com/utahnlp/weak-verifiers.git
  ```
2. Enter the project directory. Create a new virtual environment and activate it.
  ```console
  foo@bar:~$ cd weak-verifiers
  foo@bar:prompts-for-structures$ python -m venv <venv_name>
  foo@bar:prompts-for-structures$ source activate <venv_name>/bin/activate
  (<venv_name>)foo@bar:weak-verifiers$
  ```
3. Create necessary data and dump folders.
  ```console
  (<venv_name>)foo@bar:weak-verifiers$ mkdir -p models
  ```
4. Install package requirements.
  ```console
  (<venv_name>)foo@bar:weak-verifiers$ pip install -r requirements.txt
  ```
5. Install gurobipy to run DirectProbe. Install the Gurobi Optimzer (https://support.gurobi.com/hc/en-us/articles/4534161999889-How-do-I-install-Gurobi-Optimizer-). You'll need a Gurobi licence to use the optimizer. If you are in academia, you can obtain one at: https://www.gurobi.com/academia/academic-program-and-licenses/
  ```console
  (<venv_name>)foo@bar:weak-verifiers$ pip install gurobipy
  ```
6. Install [DirectProbe](https://github.com/utahnlp/DirectProbe/tree/main) and its relevant requirements. The DirectProbe project directory should reside the in the `scripts` folder.


## Gujarati SNACS Dataset
As a part of this paper, we contribute the first semantically annotated dataset in Gujarati which annotates adpositional and case supersenses according to the SNACS schema. Gujarati SNACS contains supersense annotations for all adpositions and case markers present in the freely available Gujarati translation by Dr. Sulbha Natraj of the popular childrens' book _Le Petit Prince_ (The Little Prince) by Antoine de Saint-Exupéry. The translation had to be digitized and is available under `data/nanakdo_rajkumar.txt`.

The annotated data is available under `data/annotated/` where it has been divided into chapter files. Please head on to `data/README.md` for details regarding the format.


 ## Citation
 Citation is to be updated soon.

 ## Contact
 Please open an issue if you find trouble running our codebase. <br>
 In case the issue is not acknowledged/addressed within a week, please email at maitrey.mehta@utah.edu
 
