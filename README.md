<h1 align="center">
自动驾驶算法
</h1>
<p align="center">
自动驾驶相关算法代码仓库，Python 语言版。
</p>
<p align="center">
简体中文 | <a href="README_en.md">English</a>
</p>

[![GitHub Repo stars](https://img.shields.io/github/stars/muziing/PythonAutomatedDriving)](https://github.com/muziing/PythonAutomatedDriving)
![License](https://img.shields.io/github/license/muziing/PythonAutomatedDriving)

[![Poetry](https://img.shields.io/endpoint?url=https://python-poetry.org/badge/v0.json)](https://python-poetry.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Checked with mypy](https://www.mypy-lang.org/static/mypy_badge.svg)](https://mypy-lang.org/)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/muziing/PythonAutomatedDriving/main.svg)](https://results.pre-commit.ci/latest/github/muziing/PythonAutomatedDriving/main)

## 简介

本代码仓库收录若干关于「自动驾驶-规划与控制算法」相关的代码，全部由 Python 语言实现。希望能为其他 PnC 算法学习者提供一批接口清晰统一、可直接复用的代码段，避免减少重复造轮子。

## 如何使用

### 一、获取源码

1. Star 本仓库 :smiley:
2. 通过以下方式之一获取源码：
    - 下载代码压缩包：<https://github.com/muziing/PythonAutomatedDriving/archive/refs/heads/main.zip>
    - （仅需要参与开发时）克隆仓库 `git clone https://github.com/muziing/PythonAutomatedDriving.git`
3. 进入项目目录 `cd /your/path/to/PythonAutomatedDriving`

### 二、配置虚拟环境与安装依赖

**方式 A**： venv 与 pip

1. 确保 Python 版本与 [pyproject.toml](./pyproject.toml) 中要求的一致
2. 创建虚拟环境
    - Windows: `python -m venv --upgrade-deps venv`
    - Linux/macOS: `python3 -m venv --upgrade-deps venv`
3. 激活虚拟环境
    - Windows: `venv\Scripts\activate`
    - Linux/macOS: `. venv/bin/activate`
4. 安装依赖 `pip install -r requirements.txt`

**方式 B**：[Poetry](https://python-poetry.org)

1. 确保 Python 版本与 [pyproject.toml](./pyproject.toml) 中要求的一致
2. 按[官方文档](https://python-poetry.org/docs/#installation)指示安装 Poetry
3. 创建虚拟环境：`poetry env use /full/path/to/python`（注意替换路径，使用符合版本要求的解释器）
4. 安装依赖：`poetry install --no-dev`
5. 使用该虚拟环境： `poetry shell`（或在 PyCharm 等 IDE 中配置）

**方式 C**：[Anaconda](https://www.anaconda.com/)

由于主要开发与测试工作仅在 Poetry 环境中进行，conda 环境可能出现意料之外的错误。请[提交 issue](https://github.com/muziing/PythonAutomatedDriving/issues) 帮助我发现和解决这些问题。

````shell
conda env create -f environment.yml
````

### 三、运行！

在每个子项目目录中运行 Python 脚本。

## 目录

### [utilities](src/utilities) 辅助函数

- [coordinate_transformation](src/utilities/coordinate_transformation.py) - 坐标变换
- [discrete_curve_funcs](src/utilities/discrete_curve_funcs.py) - 处理二维离散曲线的函数
- [numpy_additional_funcs](src/utilities/numpy_additional_funcs.py) - 一些补充numpy功能的自实现函数

### [LocalPathPlanning](src/LocalPathPlanning) - 局部路径规划

- [FSAE Path Planning](src/LocalPathPlanning/FSAE_PathPlanning) - 一种适用于 FSAE 无人赛车高速循迹项目的路径规划算法

## 许可协议

````text
Copyright 2022-2023 muzing

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
````
