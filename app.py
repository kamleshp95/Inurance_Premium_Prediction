from flask import Flask, render_template,url_for,request
from inference import get_prediction
app = Flask(__name__)

@app.route("/")
def home():
    sex = ["Male","Female"]
    smoker = ["Yes","No"]
    region = ['southwest', 'southeast', 'northwest', 'northeast']
    return render_template("index.html",gender=sex,smoker=smoker,region=region)

@app.route('/predict', methods=['POST'])
def predict():
    age = int(request.form.get('age'))
    sex = request.form.get('sex').lower()
    bmi = float(request.form.get('bmi'))
    children = int(request.form.get('children'))
    smoker = request.form.get('smoker')
    region = request.form.get('region')
    print( age, sex, bmi,children,smoker, region)
    prediction= get_prediction( age, sex, bmi,children,smoker, region)
    

    return render_template("index.html",prediction=prediction)
if __name__ == "__main__":
    app.run(debug=True)
