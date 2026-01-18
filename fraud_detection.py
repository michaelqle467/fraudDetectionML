import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Loading the model
model = joblib.load("fraud_detection_pipeline.pkl")

st.set_page_config(
    page_title="Fraud Detection App",
    layout="wide"
)

#side bars
st.sidebar.title("Transaction Input")

transaction_type = st.sidebar.selectbox(
    "Transaction Type",
    options=["PAYMENT", "TRANSFER", "CASH_OUT", "DEPOSIT"]
)

amount = st.sidebar.number_input("Amount", min_value=0.0, value=1000.0, step=1.0)
oldbalanceOrg = st.sidebar.number_input("Old Balance (Sender)", min_value=0.0, value=10000.0, step=1.0)
newbalanceOrig = st.sidebar.number_input("New Balance (Sender)", min_value=0.0, value=9000.0, step=1.0)
oldbalanceDest = st.sidebar.number_input("Old Balance (Receiver)", min_value=0.0, value=0.0, step=1.0)
newbalanceDest = st.sidebar.number_input("New Balance (Receiver)", min_value=0.0, value=0.0, step=1.0)

#landing

st.title("Fraud Detection App")

input_data = pd.DataFrame({
    "type": [transaction_type],
    "amount": [amount],
    "oldbalanceOrg": [oldbalanceOrg],
    "newbalanceOrig": [newbalanceOrig],
    "oldbalanceDest": [oldbalanceDest],
    "newbalanceDest": [newbalanceDest]
})

st.subheader("Transaction Details")
st.table(input_data)

if st.button("Predict Fraud"):
    prediction = model.predict(input_data)
    proba = model.predict_proba(input_data)[0][1]  # probability of fraud

    st.subheader("Prediction Result")
    st.write("Fraud Probability: {:.2%}".format(proba))

    if prediction[0] == 1:
        st.write("Prediction: FRAUD")
    else:
        st.write("Prediction: LEGIT")

   
    # Graph 1: Probability bar
   
    st.subheader("Probability Distribution")
    fig1, ax1 = plt.subplots()
    sns.barplot(
        x=["Legit", "Fraud"],
        y=[1 - proba, proba],
        ax=ax1
    )
    ax1.set_ylim(0, 1)
    ax1.set_ylabel("Probability")
    ax1.set_title("Prediction Probability")
    st.pyplot(fig1)

   
    # Graph 2: Amount vs Fraud risk

    st.subheader("Amount vs Fraud Risk (Simulated)")
    fig2, ax2 = plt.subplots()
    sns.scatterplot(
        x=[amount],
        y=[proba],
        size=[amount],
        sizes=(50, 500),
        ax=ax2
    )
    ax2.set_ylim(0, 1)
    ax2.set_xlabel("Transaction Amount")
    ax2.set_ylabel("Fraud Probability")
    ax2.set_title("Amount vs Fraud Probability")
    st.pyplot(fig2)

  
   # Transaction type fraud risk
 
    st.subheader("Type-Based Risk (Example)")
    type_risk = pd.DataFrame({
        "type": ["PAYMENT", "TRANSFER", "CASH_OUT", "DEPOSIT"],
        "risk": [0.02, 0.08, 0.05, 0.01]
    })

    fig3, ax3 = plt.subplots()
    sns.barplot(x="type", y="risk", data=type_risk, ax=ax3)
    ax3.set_ylim(0, 0.15)
    ax3.set_xlabel("Transaction Type")
    ax3.set_ylabel("Estimated Fraud Risk")
    ax3.set_title("Estimated Fraud Risk by Transaction Type")
    st.pyplot(fig3)
