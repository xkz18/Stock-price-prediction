# Stock-price-prediction
A "window" time series method based on machine learning models was brought up to predict stock return and achieved faster calculations than ARMA model
## Table of Contents
**Dataset**
In this project, datasets are taken from a Kaggle contest sponsored by Winton Capital (Win). The datasets contain various market related data, and the goal is to forecast future unseen intraday and daily returns. The dataset has 40,000 observations (stocks), each of which includes 25 features, 2 intraday return ratios, 179 intraday returns. 

**Preprocessing**
The size of original data is over 70MB and it is full of noise. The sources of noise are both from the original data and the filling values from the last step. We use SVD method to transform the original huge matrix into a product of three small matrices.

**Build Models**

**Input files**

**utrain.dat :** Training data from SVD reconstruction, U

**vtrain.dat :** Training data from SVD reconstruction, V

**sgmatest.dat :** Test data for from SVD reconstruction, Sigma

The machine learning models were built as shown in Stock_price_prediction.ipynb

