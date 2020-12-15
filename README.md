# exoplanet_search

## General info
Repository for the 3rd project in FYS-STK 4155. The data used in this project is from The data is taken from https://www.kaggle.com/keplersmachines/kepler-labelled-time-series-data. Note that the plots in visuals/ may differ from those in report.
 
## How to Run Code

### Unzip data
go to * [data](data) folder and unzip "archive.zip" so that the data folder contains two csv files: "exoTrain.csv" and "exoTest.csv" 

### prepare pipenv (python 3.7)
* install pipenv on your system 
* clone the repository
*  in correct folder, type:
```
install pipenv
```
* enter shell:
```
pipenv shell
```
* run code file as normal

## Table of contents
* [Visuals](visuals)
* [Code](code)
* [Example runs](code/example_runs)
* [Report](report)


## example use 

## test of code/benchmarks
to test the function write
```
pytest -v 
```
or run the program normally. this also serves as benchmarks to check that each part of the code is running as expected.
