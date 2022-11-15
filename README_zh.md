<h1 align="center">
自动驾驶算法
</h1>
<p align="center">
自动驾驶相关算法代码仓库，Python 语言版。
</p>
<p align="center">
<a href="README.md">English</a> | 简体中文
</p>

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
3. 创建虚拟环境：`poetry env use /full/path/to/python`（注意替换路径）
4. 安装依赖：`poetry install --no-dev`
5. 使用该虚拟环境： `poetry shell`（或在 PyCharm 等 IDE 中配置）

**方式 C**：[Anaconda](https://www.anaconda.com/)

````shell
conda env create -f requirements/environment.yml
````

### 三、运行！

在每个子项目目录中运行 Python 脚本。

## 目录

### [AuxiliaryFunctions](AuxiliaryFunctions) 辅助函数

- [curvature](AuxiliaryFunctions/curvature.py)  计算曲率

### [LocalPathPlanning](LocalPathPlanning) 局部路径规划

- [FSAE Path Planning](LocalPathPlanning/FSAE_PathPlanning)  适用于 FSAE 无人赛车高速循迹项目的路径规划算法

## 许可协议

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
