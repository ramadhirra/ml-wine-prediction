import streamlit as st
import pickle
import numpy as np
model = pickle.load(open(r'C:\Users\User\Downloads\Random_Forest_Wine_Classifier.pkl', 'rb'))

def predict_values(flav, proline, od, color):
    # Create the input array
    input_data = np.array([[flav, proline, od, color]])

    # Perform prediction using the loaded model
    prediction = model.predict(input_data)

    return prediction

def main():
    st.title("Wine Class Prediction")
    html_temp = """
    <div style="background:#025246 ;padding:10px">
    <h2 style="color:white;text-align:center;">Wine Class Prediction ML App</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    # Input form for user input
    flav = st.number_input("Flav:")
    proline = st.number_input("Proline:")
    od = st.number_input("OD:")
    color = st.number_input("Color:")

    if st.button("Predict"):
        output = predict_values(flav, proline, od, color)
        st.success(f"The predicted wine class is {output}")

if __name__ == "__main__":
    main()
