import streamlit as st
import joblib
import pandas as pd

st.write("# Small Loan Prediction")

col1, col2, col3 = st.columns(3)

age = col1.number_input("年齡:")
sex = col2.selectbox("性別:",["M", "F"])
region = col3.selectbox("地區:",["INNER_CITY", "TOWN", "SUBURBAN", "RURAL"])
income = col1.number_input("收入:")
married = col2.selectbox("是否結婚?",["YES","NO"])
children = col3.selectbox("小孩個數?",[0,1,2,3])
car = col1.selectbox("是否有車?",["YES","NO"])
save_act = col2.selectbox("是否有活儲帳戶?",["YES","NO"])
current_act = col3.selectbox("是否有支存帳戶?",["YES","NO"])
mortgage = col1.selectbox("是否有房貸?",["YES","NO"])

df_pred = pd.DataFrame([[age,sex,region,income,married,children,car,save_act,current_act,mortgage]],
                       columns= ['age','sex','region','income','married','children','car','save_act','current_act','mortgage'])

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

        