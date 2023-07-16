import pandas as pd

def find_dataset_statistics(dataset:pd.DataFrame,target_col:str) -> tuple[int,int,int,int,float]:
    # TODO: Write the necessary code to generate the following dataset statistics given a dataframe
    # and a target column name. 

    dim = dataset.shape

    #Total number of records
    n_records = dim[0]
    #Total number of columns 
    n_columns = dim[1]
    #Number of records where target is negative
    n_negative = len(dataset[dataset[target_col] == 0])
    #Number of records where where target is positive
    n_positive = len(dataset[dataset[target_col] == 1])
    # Percentage of instances of positive target value
    if n_records == 0:
        perc_positive = 0
    else:
        perc_positive =  (n_positive / n_records)*100

    return n_records,n_columns,n_negative,n_positive,perc_positive