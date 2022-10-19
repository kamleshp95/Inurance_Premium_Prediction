import pickle

import streamlit as st
from streamlit_lottie import st_lottie
import requests
import base64
from PIL import Image

st.set_page_config(page_title='Premium Prediction', page_icon=':family:', layout='wide')



def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


lottie_coding = load_lottieurl('https://assets9.lottiefiles.com/packages/lf20_qn2yhgln.json')

loaded_model = open("gradient_boosting_regressor_model.pkl", "rb")
predictor = pickle.load(loaded_model)


def welcome():
    return "Welcome All To Insurance Premium Predictor"

image = Image.open('./static/health_insurance.jpg')
st.image(image)
def predict_premium(age, sex, bmi, children, smoker, region):
    prediction = predictor.predict([[age, sex, bmi, children, smoker, region]])
    print(prediction)
    return prediction



with st.container():
    st.write("---")
    left_column, right_column = st.columns(2)
    with left_column:
        st.title("Welcome All To Health Insurance Premium Predictor")
        html_temp = """
        <div style="background-color:black;padding:10px">
        <h2 style="color:green;text-align:center;">Health Insurance Premium Prediction</h2>
        </div>
        """
    with right_column:
        st_lottie(lottie_coding, height=300, key="coding")

    st.markdown(html_temp, unsafe_allow_html=True)
    age = st.number_input("age", min_value=0, max_value=100)

    sex = st.selectbox("sex", ('female', 'male'))

    bmi = st.number_input("bmi", min_value=0.0, max_value=50.0)

    children = st.selectbox("children", ('0', '1', '2', '3', '4', '5'))

    smoker = st.selectbox("smoker", ('yes', 'no'))

    region = st.selectbox("region", ('northwest', 'northeast', 'southwest', 'southeast'))

    result = ""
    if st.button("Predict"):
        result = predict_premium(age, sex, bmi, children, smoker, region)
    st.success('The Predicted Health Insurance Premium is {}'.format(result))
    if st.button("About"):
        st.text("Lets LEarn")
        st.text("Built with Streamlit")

if __name__ == '__main__':
    print('Predicted')
    