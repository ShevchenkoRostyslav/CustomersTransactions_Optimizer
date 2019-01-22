# CustomersTransactions_Optimizer

Data Analysis Coding Assignment

Four tasks were provided for a home assignment(see coding_assignment.pdf). 
Each of them is solved separately in a corresponding Jupyter Notebooks:
* Task1.ipynb - data extraction and train/validation split;
* Task2.ipynb - feature derivation;
* Task3and4.ipynb - Implementation of the CustomFunction class to compute and minimize a given objective function.

To <b>productionalize</b> the routine the project was also re-written in terms of separate classes:
* CustomerDataContainer (sample/CustomerDataContainer.py) is responsible for the handling of the input data, derivation of the required for the fit model features, train/validation splitting etc.
* CustomFunction (sample/CustomFunction.py) is responsible for the implementation of the given objective function and its minimization.

To run: python3 main.py --input_file='csv/all_transactions.csv' --split_date_b=19970101 --split_date_e=19971001

A couple of <b>unit tests</b> for the CustomerDataContainer class are at tests/CustomerDataContainer_tests.py

<h3>Possible improvements:</h3>
<bf>CustomerDataContainer:</bf> 

* Better solution for the training/validation split, i.e. do not overwrite the full dataset with the training part (it was a dirty quick solution)
* Implement a check for the beginning and the end of the train/validation split, namnely: whether it within the given period of time, positive, proper format etc...

<bf>CustomFunction:</bf>
* Write unit tests

<bf>Productionalization:</bf>
* Make a simple Flask-API 
