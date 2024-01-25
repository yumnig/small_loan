import streamlit as st
import joblib
import pandas as pd
import numpy as np

st.write("# Small Loan Prediction")

small_loan = pd.read_csv('small_loan.csv')

col1, col2, col3 = st.columns(3)

all_col = []

for i, col in enumerate(small_loan.columns[1:-1]):
    if i % 3 == 0:
        if small_loan.dtypes[col] == 'object':
            all_col.append(col1.selectbox(col,list(small_loan[col].unique())))
        else:
            all_col.append(col1.number_input(col))
    elif i % 3 == 1:
        if small_loan.dtypes[col] == 'object':
            all_col.append(col2.selectbox(col,list(small_loan[col].unique())))
        else:
            all_col.append(col2.number_input(col))
    else:
        if small_loan.dtypes[col] == 'object':
            all_col.append(col3.selectbox(col,list(small_loan[col].unique())))
        else:
            all_col.append(col3.number_input(col))
            
df_pred = pd.DataFrame(np.array(all_col).reshape(1,-1), columns=small_loan.columns[1:-1])

df_pred.replace({'YES':1, 'NO':0, 'M': 1, 'F': 0}, inplace=True)
df_pred = pd.get_dummies(df_pred, prefix=['region', 'children'], prefix_sep='_', columns=['region', 'children'])

model = joblib.load('small_loan_rf_model.pkl')

# Need this part, otherwise the number of attributes are not consistent with the model
missing_cols = [c for c in model.feature_names_in_ if c not in df_pred.columns]
for c in missing_cols:
    df_pred[c] = 0
    
df_pred = df_pred[model.feature_names_in_]

prediction = model.predict(df_pred)
prediction_prob = model.predict_proba(df_pred)

if st.button('預測'):
    if(prediction[0]==0):
        st.write('<p class="big-font">此人對小額信貸<font color="red">沒有興趣</font>.</p>',unsafe_allow_html=True)
    else:
        st.write('<p class="big-font">此人對小額信貸<font color="blue">有興趣</font>.</p>',unsafe_allow_html=True)
        
    st.write('<p class="big-font">'+str(prediction_prob)+'</p>',unsafe_allow_html=True)    

        