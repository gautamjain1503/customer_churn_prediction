from flask import Flask, flash, request, redirect, url_for, render_template
import urllib.request
import os
from customer_churn_prediction.pipeline.stage_3_predict import Predict

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads/'
app.secret_key = "secret_base_key"


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/', methods=['GET','POST'])
def validator():
    if request.method=='POST':
        dictionary={}
        dictionary["creditscore"]=int(request.form["creditscore"])
        dictionary["age"]=int(request.form["age"])
        dictionary["tenure"]=int(request.form["tenure"])
        dictionary["balance"]=float(request.form["balance"])
        dictionary["numberofproducts"]=int(request.form["numberofproducts"])
        dictionary["estimatedsalary"]=float(request.form["estimatedsalary"])
        dictionary["geography"]=request.form["geography"]
        dictionary["gender"]=request.form["gender"]
        dictionary["hascrcard"]=int(request.form["hascrcard"])
        dictionary["isactivemember"]=int(request.form["isactivemember"])
        print(dictionary)
        obj = Predict()
        result=obj.main(dictionary=dictionary)
        print(result)
        return render_template('index.html', result=result)

    return render_template('index.html')


@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == "__main__":
    app.run(host= '0.0.0.0', debug=False)