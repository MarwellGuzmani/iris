import streamlit as st
import pickle
import numpy as np

# Load models
lin_model = pickle.load(open('lin_model.pkl', 'rb'))
log_model = pickle.load(open('log_model.pkl', 'rb'))
svc_model = pickle.load(open('svc_model.pkl', 'rb'))

# Classify prediction output
def classify(num):
    if num < 0.5:
        return 'Setosa'
    elif num < 1.5:
        return 'Versicolor'
    else:
        return 'Virginica'

def main():
    st.title("Iris Classification")

    # Header
    html_temp = """
    <div style="background-color:#f19c9ce6; padding:10px margin-bottom:20px;">
    <h2 style="color:white;text-align:center;">Iris Classification</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    # Model selection
    activities = ['Linear Regression', 'Logistic Regression', 'SVM']
    option = st.sidebar.selectbox('Which model would you like to use?', activities)
    st.subheader(option)

    # Input text boxes
    try:
        sl = float(st.text_input("Enter Sepal Length (cm)", "5.1"))
        sw = float(st.text_input("Enter Sepal Width (cm)", "3.5"))
        pl = float(st.text_input("Enter Petal Length (cm)", "1.4"))
        pw = float(st.text_input("Enter Petal Width (cm)", "0.2"))

        inputs = [[sl, sw, pl, pw]]

        # Predict and show result
        if st.button('Classify'):
            if option == 'Linear Regression':
                result = lin_model.predict(inputs)
                st.success(f"Predicted: {classify(result)}")
            elif option == 'Logistic Regression':
                result = log_model.predict(inputs)
                st.success(f"Predicted: {classify(result)}")
            else:
                result = svc_model.predict(inputs)
                st.success(f"Predicted: {classify(result)}")
    except ValueError:
        st.warning("Please enter valid numeric values for all input fields.")

if __name__ == '__main__':
    main()
