# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pickle
import scipy

def convertPKLtoMAT(): 
    # The path to your pickle file
    pickle_file_path = 'emg' + '.pkl'
    # Open the file in binary read mode
    with open(pickle_file_path, 'rb') as file:
        # Load the data from the file
        emg = pickle.load(file)
    
    # Save the dictionary to a .mat file
    scipy.io.savemat('emg.mat', emg)
    
    