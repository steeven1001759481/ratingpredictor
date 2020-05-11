from flask import Flask ,render_template ,request
import numpy as np
from sklearn.externals import joblib
app = Flask(__name__)

@app.route("/",methods=['POST','GET'])
def index():
    if request.method == 'POST':
        x=request.form['review']
        sam=[x]
        vector = joblib.load('vector.sav')
        ridge = joblib.load('ridge_model.sav')
        input_v = vector.transform(sam)
        result = np.round(ridge.predict(input_v))
        result=str(result).strip('[]')
        return render_template('index.html',result=result)
    else:
        return render_template('index.html')

if __name__ == "__main__":
    app.run()