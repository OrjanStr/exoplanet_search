# exoplanet_search

## General info
Repository for the 3rd project in FYS-STK 4155. The data used in this project is from The data is taken from https://www.kaggle.com/keplersmachines/kepler-labelled-time-series-data. Note that the plots in visuals/ may differ from those in report.
 
## How to Run Code

### Unzip data
go to * [data](data) folder and unzip "archive.zip" so that the data folder contains two csv files: "exoTrain.csv" and "exoTest.csv" or download the data directly from kaggle.

### pipenv
We were not able to create a pipenv because of some complications with our computers and the python installation. 

## Table of contents
* [Visuals](visuals)
* [Code](code)
* [Example runs](code/example_runs)
* [Test_runs](test_runs)
* [Report](report)


## example use 

## test of code/benchmarks
to test the function, while in [Test runs](test_runs), write
```
pytest -v 
```
or run the program normally. this also serves as benchmarks to check that each part of the code is running as expected.
The parts of the code implemented with scikit Learn does not have tests as this package is extensively tested already.
