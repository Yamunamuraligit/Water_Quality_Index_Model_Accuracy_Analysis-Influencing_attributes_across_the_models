import streamlit as st 

st.title("Data visualisation")

st.sidebar.success("select a above page")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.info("XGBoostClassifier")
Test = st.session_state['xgb_test_accuracy']
Test_1 = st.session_state['xgb_test_accuracy_1']
Test_2 = st.session_state['xgb_test_accuracy_2']
Test_3 = st.session_state['xgb_test_accuracy_3']
models = pd.DataFrame({'Models':['10_Attr','7_Attr','5_Attr','3_Attr'],
                        'Values': [Test,Test_1,Test_2,Test_3]})
models.set_index('Models', inplace=True)
st.bar_chart(models)

st.info("Logistic Regression")

Test = st.session_state['logistic_test_accuracy']
Test_1 = st.session_state['logistic_test_accuracy_1']
Test_2 = st.session_state['logistic_test_accuracy_2']
Test_3 = st.session_state['logistic_test_accuracy']
models = pd.DataFrame({'Models':['10_Attr','7_Attr','5_Attr','3_Attr'],
                        'Values': [Test,Test_1,Test_2,Test_3]})
models.set_index('Models', inplace=True)
st.bar_chart(models)

st.info("Artificial Neural Network (ANN) Model")

Test = st.session_state['ann_test_accuracy']
Test_1 = st.session_state['ann_test_accuracy_1']
Test_2 = st.session_state['ann_test_accuracy_2']
Test_3 = st.session_state['ann_test_accuracy_3']
models = pd.DataFrame({'Models':['10_Attr','7_Attr','5_Attr','3_Attr'],
                        'Values': [Test,Test_1,Test_2,Test_3]})
models.set_index('Models', inplace=True)
st.bar_chart(models)
