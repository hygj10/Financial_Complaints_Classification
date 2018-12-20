# financial_complaints_classification
Intro to ML Final Project

Customer service is vital in many industries. Helping and responding to your consumers swiftly and accurately plays a crucial role in retaining customers by keeping them saitsfied and informed. 
One way of increasing speed and efficiency in dealing with these situations is to have a program that immediately classifies a type of complaint or inquiry of a consumer and directly directs them to experts in respective areas. This saves time for both the customers and the businesses by getting rid of an entire layer of communication.

In this notebook, different models are used and compared in solving the problem of classifying financial complaints from the US Consumer Finance Complaints available on Kaggle. Parts were inspired/brought from works of Susan Li (https://towardsdatascience.com/@actsusanli) and zqhZY (https://github.com/zqhZY?tab=repositories) and modified to different extents to either fit the problem or (to attempt) to enhance the results.

The evaluation results for the CNN and LSTM models are still in process as they are taking quite some time and I wasn't able to set up a suitable GPU to train these yet. Results will be updated when done.

* final_project2.ipynb - Jupyter Notebook with all models that comes along with descriptions, details, and thoughts.
* extra_cleaning.ipynb - Some extra/experimental data cleaning/wrangling for different uses.
* cnn_lstm_.py - CNN and LSTM code
* logistic_doc2vec.py - logistic regression + doc2vec code
* complaints_cleaned.csv - cleaned dataset of consumer complaints
* complaints_cleaned_label.csv - cleaned dataset labelled with numbers for each product type

WARNING: The CNN model and the LSTM model displays the accuracy of each batch, printing an extremely long text. 

The original Consumer Complaints File can be found in Kaggle here: https://www.kaggle.com/cfpb/us-consumer-finance-complaints
The size exceeds 25mb so it cannot be uploaded here. However, two 'cleaned' versions of the dataset are available in the repository as zip files (and should probably be unzipped if the Jupyter notebook is to be run).

UPDATES MAY COME.

Update: due to my computer heating up, there might not be final results on the LSTM soon, but based on the batch number and accuracy compared to the respective ones in the CNN model, the LSTM ones' accuracy seem to be lower. This may have been the result of mishandling pre-trained word embeddings (not being able to integrate them well with the LSTM) or mismatch.
I have been trying to run this on hpc prince but there seem to be some problems with load modules.
