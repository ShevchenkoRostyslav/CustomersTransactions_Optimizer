import argparse
from sample.CustomerDataContainer import CustomerDataContainer
from sample.CustomFunction import CustomFunction

def ParseUserInput():
    """Function to parse user input.
    """
    parser = argparse.ArgumentParser("Fit the model to customer data")
    parser.add_argument('--input_file', required=True, help='Input .csv file with customer data')
    parser.add_argument('--split_date_b', required=True, type=int, help='Training-Validation split: begin of the training part YYYYMMDD')
    parser.add_argument('--split_date_e', required=True, type=int, help='Training-Validation split: end of the training part YYYYMMDD')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    # Parse user input
    user_args = ParseUserInput()
    # user input file
    input_file = user_args.input_file
    # begin of the training data period
    split_date_b = user_args.split_date_b
    # end of the training data period
    split_date_e = user_args.split_date_e

    # Prepare the input 
    container = CustomerDataContainer(input_file)
    # train validation split
    df_train, df_val = container.Train_Val_split(split_date_b,split_date_e,use_train=True)
    # compute required features
    container.compute_x()
    container.compute_tx_T(split_date_e)
    # save processed data to csv file
    container.save_to_csv(filename='csv/summary_customers.csv')
    
    # Fit the model
    # get the DataFrame 
    df = container.get_fit_df()
    # instantiate a model 
    model = CustomFunction(verbose=False)
    # Fit with the custom minimization routine
    model.fit(df,minimizer_strategy=[(2,'Nelder-Mead')])
    model.print_final_results()
    # save results to the csv file
    model.saveParameters('csv/estimated_parameters.csv',digits=2)
    