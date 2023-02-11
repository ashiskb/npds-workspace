# Deep Learning Neural Network Derivation and Testing to Distinguish Acute Poisonings
* Code repository

## List of files (in toplogical order)
1. ``DNN_Poisoning_detector_Keras.ipynb`` : 
    - A jupyter notebook on preprocessing, EDA of dataset, development and experimenting with the proposed DNN algorithm using Tensorflow 2.0 (+Keras) compute framework.
2. ``DNN_Poisoning_detector_PyTorch.ipynb`` : 
    - A jupyter notebook on preprocessing, EDA of dataset, development and experimenting with the proposed DNN algorithm using PyTorch compute framework.
3. ``indVal-ashiskb-2-npds-data.R`` :
    - An R script for indicator value (INDVAL) calculation on the given dataset. It saves the scores as a csv file that will be read by the jupyter notebook listed next.
4. ``DNN_Poisoning_detector_INDVAL+Keras.ipynb`` :
    - A jupyter notebook that includes DNN model training on selected number of features according to the INDVAL analysis.
    - 
5. ``DNN_Poisoning_detector_INDVAL+Keras+SHAP.ipynb`` and ``*.py`` :
    - A jupyter notebook to ensure explainability in our DNN model prediction using SHAP. It is advised not to execute directly the jupyter notebook due to the memory requirement by the ``SHAP.shap_values()`` function. Instead, save it into a python script and run it in shell for achieving optimized performance.