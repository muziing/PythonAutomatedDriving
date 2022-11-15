<h1 align="center">
Automated driving algorithms
</h1>
<p align="center">
Python codes for automated driving algorithms.
</p>
<p align="center">
English | <a href="README_zh.md">简体中文</a>
</p>

## How to use

### 1. Get source code

1. Add star to this repo if you like it :smiley:
2. Get source code by one of the following methods
    - [Download code zip](https://github.com/muziing/PythonAutomatedDriving/archive/refs/heads/main.zip)
    - (For development only) clone repo `git clone https://github.com/muziing/PythonAutomatedDriving.git`
3. Go to the project directory `cd /your/path/to/PythonAutomatedDriving`

### 2. Creating Virtual Environment and Install requirements

**Option A**: venv & pip

1. Ensure that the Python version matches the one required in [pyproject.toml](./pyproject.toml).
2. Create virtual environment
    - Windows: `python -m venv --upgrade-deps venv`
    - Linux/macOS: `python3 -m venv --upgrade-deps venv`
3. Activate virtual environment
    - Windows: `venv\Scripts\activate`
    - Linux/macOS: `. venv/bin/activate`
4. Install the requirements `pip install -r requirements.txt`

**Option B**: [Poetry](https://python-poetry.org)

**Option C**: [Anaconda](https://www.anaconda.com/)

````shell
conda env create -f requirements/environment.yml
````

### 3. Run!

Execute Python script in each directory.

## Table of Contents

### [AuxiliaryFunctions](AuxiliaryFunctions)

- [curvature](AuxiliaryFunctions/curvature.py)  calculate curvature

### [LocalPathPlanning](LocalPathPlanning)

- [FSAE Path Planning](LocalPathPlanning/FSAE_PathPlanning)  适用于 FSAE 无人赛车高速循迹项目的路径规划算法

## License

````text
Copyright 2022 muzing

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
````
