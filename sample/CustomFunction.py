#Import packages
import pandas as pd
import numpy as np
import scipy as sp
# treatment of the warnings
import warnings
# ln of the Gamma function 
from scipy.special import gammaln as lnG
# for minimization
from scipy import optimize
# write the final parameters
import csv

class CustomFunction:
    def __init__(self, verbose=False):
        """Constructor + initialization of the fit parameters
        
        Arguments:
        verbose - for more elaborative output (default: False)
        """
        self.verbose = verbose
        #initialize model parameters
        self.__initialize_par()
        
    def __obj_f(self, params, df):
        """Implementation of the objective function.
        NLL = -1/N Sum( lnA1i + lnA2i + ln( exp(lnA3i) + delta exp(lnA4i)) )

        Argumentsa:
        params - set of the function parameters (r,alpha,a,b)
        df - DataFrame with the input data
        """
        #update parameters
        self.__update_params(params)
        
        #normalise alpha
        alpha_norm =self.alpha * self.norm
        # protection from negative parameters
        if self.r <=0 or self.b <=0 or self.a <= 0 or alpha_norm <= 0:
            return np.inf
        # number os samples
        N = len(df)
        # loss function
        with warnings.catch_warnings():
            # catch warnings from "bad" values in the log. 
            # they are treated with the corresponding value of the NLL
            warnings.simplefilter("ignore",RuntimeWarning)
            obj_i = self.__loss(df,self.r,alpha_norm,self.a,self.b)
        # protection from the inf values that might arise from division by zero
        if obj_i.isin([np.inf]).values.any():
            return np.inf
        # normalised NLL
        out = obj_i.sum() * (-1) / N
        return out
    
    def __loss(self, df,r,alpha,a,b):
        """Implementation of the loss function.

        Arguments:
        df - DataFrame with the input data
        r, alpha, a ,b - model parameters 
        """
        lnA1 = lnG(r+df.x) + r*np.log(alpha) - lnG(r)
        lnA2 = lnG(a+b) + lnG(b+df.x) - lnG(b) - lnG(a+b+df.x)
        lnA3 = -(r+df.x)*np.log(alpha+df['T'])
        lnA4 = np.log(a) - np.log(b+df.x-1) - (r+df.x)*np.log(alpha + df.tx)
        deltaA4 = self.__deltaA4(df.x,np.exp(lnA4))
        out = lnA1 + lnA2 + np.log(np.exp(lnA3) + deltaA4)
        return out
    
    def __deltaA4(self,x,A4):
        """Implementation of the delta function.
        deltaX = 0 if x <= 0 
        deltaX = X otherwise

        Arguments:
        x - number of transactions
        A4 - Series with data
        """
        A4[x <= 0] = 0
        return A4
    
    def __update_params(self, new_params):
        """Update model parameters r, alpha, a ,b

        Arguments:
        new_params - set of parameters used to update the model r, alpha, a ,b
        """
        self.r = new_params[0].copy()
        self.alpha = new_params[1].copy()
        self.a = new_params[2].copy()
        self.b = new_params[3].copy()
    
    def __initialize_par(self):
        """Method to initialize parameters with Gaus(1,0.05)
        """
        self.r, self.alpha, self.a, self.b = np.random.normal(1, 0.05,4)
    
    def set_r(self, r):
        """Method to set parameter r
        """        
        if r <= 0: 
            raise ValueError('r should be positive')
        self.r = r
        
    def set_alpha(self, alpha):
        """Method to set parameter alpha
        """
        if alpha <= 0: 
            raise ValueError('alpha should be positive')
        self.alpha = alpha
    
    def set_a(self, a):
        """Method to set parameter a
        """        
        if a <= 0: 
            raise ValueError('a should be positive')
        self.a = a
        
    def set_b(self, b):
        """Method to set parameter b
        """
        if b <= 0: 
            raise ValueError('b should be positive')
        self.b = b
    
    def set_parameters(self, r, alpha, a, b):
        """Method to set the fit parameters
        """
        self.set_r(r), self.set_alpha(alpha), self.set_a(a), self.set_b(b)
    
    def fit(self, df, minimizer_strategy = [(2,'Nelder-Mead')]):
        """Implementation of the custom minimisation procedure.
        
        Arguments:
        df - pandas DataFrame with the input data of the next format (x,tx,T),
        minimizer_strategy - list of pears specifying the minimzersand number of calls to be used
        (default [(2,'Nelder-Mead')])
        """
        # check the input data
        self.__validate_df(df)
        # compute the scalefactor and normalise the data
        norm_df = self.__normalise_data(df)
        # initial parameters
        initial_pars = [self.r,self.alpha,self.a,self.b]
    
        for num_iter,minimizer in minimizer_strategy:
            print('Run {} minimizer'.format(minimizer))
            for i in range(1,num_iter+1):
                print('Step ',i)
                initial_pars = [self.r,self.alpha,self.a,self.b]
                if self.verbose: print('Initialisation: r = ',self.r, ' alpha = ',self.alpha, ' a = ',self.a, ' b = ',self.b)
                # perform the minimization
                self.fit_res = sp.optimize.minimize(self.__obj_f,x0 = initial_pars,args=(norm_df), method=minimizer) 
                # update the model parameters
                self.__update_params(self.fit_res.x)
                if i!= num_iter+1 and self.verbose:
                    self.print_final_results()
                    print('\n')
                
    def fit_global(self, df, minimizer_strategy = 'Nelder-Mead',num_iter = 100):
        """Implementation of the global (brut-force) minima finding algorithm - basinhopping
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.basinhopping.html#scipy.optimize.basinhopping
        Large number of iterations is recommended due to a step-based optimisation procedure.
        Fit succeed if 15% of num_iter the global minimum candidate remains the same.
        
        Arguments:
        df - pandas DataFrame with the input data of the next format (x,tx,T),
        minimizer_strategy - The minimization method (default: Nelder-Mead),
        num_iter - number of max iterations (default: 100)
        """
        # check the input data
        self.__validate_df(df)
        # compute the scalefactor and normalise the data
        norm_df = self.__normalise_data(df)
        # initial parameters
        initial_pars = [self.r,self.alpha,self.a,self.b]
        print('********Basin-hopping minimisation********')
        if self.verbose: print('Initialisation: r = ',self.r, ' alpha = ',self.alpha, ' a = ',self.a, ' b = ',self.b)
        # perform the minimisation
        self.fit_res = sp.optimize.basinhopping(self.__obj_f,x0 = initial_pars, 
                                            minimizer_kwargs={"method": minimizer_strategy,'args':(norm_df)},
                                            stepsize = 0.05,
                                            niter = num_iter,
                                            niter_success = int(num_iter*0.15),
                                            disp = self.verbose)
        # update the model parameters
        self.__update_params(self.fit_res.x)
    
    def __validate_df(self,df):
        """Method to check whether the input DataFrame contains
        required Series (x,xt,T) and free of NaNs.
        
        Arguments:
        df - DataFrame with the input data
        """
        cols = ['x','tx','T']
        for c in cols:
            if c not in df.columns:
                raise ValueError('Column ' + c + ' is missing in the input DataFrame.')
    
    def __normalise_data(self,df):
        """Method to normalise parameters tx and T.

        Arguments:
        df - DataFrame with the input data
        """
        df_temp = df.copy()
        self.__norm_sf(df)
        df_temp['tx']*=self.norm
        df_temp['T']*=self.norm
        return df_temp
    
    def __norm_sf(self,df):
        """Method to compute the normalisation factor
        """
        max_T = df['T'].max()
        self.norm = 10./ max_T
    
    def print_final_results(self):
        """Method to print final results.
        """
        print('*********Minimisation is done*********')
        # For some minimzers success and status are not defined
        # e.g. doesn't work for basinhopping
        try:
            print('Success: ',self.fit_res.success)
            print('Status: ',self.fit_res.status)
        except Exception: pass
        print('NLL: ',self.fit_res.fun)
        print('Optimized parameters: r = ',self.r, ' alpha = ',self.alpha, ' a = ',self.a, ' b = ',self.b)
        if self.verbose:
            print(self.fit_res)
            
    def getFitResults(self):
        """Method to return the fit results.
        """
        try:
            return self.fit_res
        except Exception:
            print('Fit has not been applied yet, no fit result object exist.')
            return None
    
    def getFitParameters(self,digits = None):
        """Method to return the fit parameters.

        Arguments:
        digits - number of digits to round the output (default: None - not round)
        """
        if digits != None:
            return (round(x,digits) for x in self.getFitParameters())
        return self.r, self.alpha, self.a, self.b

    def saveParameters(self,filename = 'csv/estimated_parameters.csv', digits = 2):
        """Save the model parameters to a .csv file.
        
        Argumentsa:
        filename - name of the output file (default: csv/estimated_parameters.csv)
        digits - number of digits to round the output (default: 2)
        """
        with open('csv/estimated_parameters.csv','w') as file:
            writer = csv.writer(file,delimiter=',',)
            writer.writerow(['r','alpha','a','b'])
            writer.writerow(self.getFitParameters(digits)) # rounded to 2 digits