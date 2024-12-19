# SDA-EEG-Group-Project
Scientific Data Analysis Group Project


- data_preprocessing.py reads for the data which is contained in folders in .eea format. It converts this into a usable pandas dataframe after performing a fast fourier transform. The intended use is calling the function preprocess() from another file which would load the data.
- correlation_matrix.py visualizes the correlation between channels using spearman's correlation. It uses ranks to determine the correlation which while showing the trend, does lead to a loss of information when it comes to actual magnitudes of power values. Hence whe used csd in further testing. To get a better look into the correlation an attempt was made to plot it ontop of a topomap. 

- The CSD_matrix script provides the "cross_spectral_matrix" & "average_csd" used in "Test2_Procedure.ipynb" & "CSD_Global_Visualization.ipynb" & "bootstrapping.ipynb"

- Running the MNE_Visualization script will first display the RAW EEG (Voltage Vs Time) of a "healthy" patient and then the Power Spectral Density Curve of that same patient. When the windows are closed, it will show you the same curves for a "schizophrenic" patient. For the Voltage Vs Time Curves showing all channels, pressing "-" and "+" on your keyboard will scale down or up the signal for better visualisation. It is also specified in the help tab of that plot.

- The "CSD_Global_Visualization" Notebook, will display the CSD Matrices & Difference Matrices from our dataset, for both groups and for all types of waves. It will also show you a mapping of the channels for better understanding the content of the Matrices.
- bootstrapping.ipynb contains the code necessary to get the p-value matrices for each wave regarding CSD. The sample size and number of iteration used is lowered to make computation faster, so whilst it may not be super accurate, it is still possible to see the general trend.
- The Test2_Procedure Notebook will walk you through the 2nd statistical analysis procedure we performed to assess how significant are the CSD differences between both groups (healthy & diagnosed schizophrenic groups)

-The 'coherence_matrix' notebook calculates the coherence of the regions (the functional connectivity) in healthy and schizophrenic patients. Because the coherence and CSD matrices we calculated looked identical, we decided to continue the analysis on the CSD values because they are computationally less heavy, and therefore would allow for easier analysis. 
-The 'AR_surrogate_shuffle' notebook calculated an autoregression model (AR) on the original data and extracted the residuals of this AR with the aim to create a surrogate data to test the null hypothesis. Residual shuffling did not result in a very different surrogate dataset, which then not result in a significant H0 testing. We therefor did not present this data, although we mentioned we tried AR H0 testing. 

