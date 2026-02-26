\# Telco Customer Churn Prediction \& Retention Simulator



This project builds an end-to-end machine learning pipeline to predict customer churn using the Telco Customer Churn dataset. The final model is deployed using Streamlit and includes a simple retention ROI simulator.



---



\## Project Overview



The goal of this project is to:



\- Explore and clean customer data

\- Compare classification models (Logistic Regression vs Gradient Boosting)

\- Evaluate performance using ROC-AUC and PR-AUC

\- Deploy the trained model using Streamlit

\- Simulate retention strategies based on churn probability



This project focuses not only on prediction accuracy but also on how model outputs can support business decisions.



---



\## Models Used



\- Logistic Regression (baseline model)

\- Gradient Boosting (final selected model)



Gradient Boosting was selected based on validation performance and its ability to capture more complex patterns in the data.



---



\## Streamlit Application



The Streamlit app allows:



\- Single customer churn prediction (CSV upload)

\- Batch scoring of multiple customers

\- Downloading scored results

\- Retention targeting simulation (Top-k or Threshold strategy)

\- ROI estimation (expected saved value vs cost)



To run locally:



pip install -r requirements.txt  

python train\_and\_save.py  

streamlit run app.py



---



\## Dataset



The dataset is included in the repository for testing purposes.



Location:

data/WA\_Fn-UseC\_-Telco-Customer-Churn.csv



---



\## Author



Kerem Ozguler  

Berlin, Germany

