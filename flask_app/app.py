from flask import Flask, render_template, request
import pandas as pd

from src.pipeline.predict_pipe import PredictPipeline

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    result = None

    if request.method == "POST":
        data = {
            "LIMIT_BAL": float(request.form["LIMIT_BAL"]),
            "AGE": int(request.form["AGE"]),

            "SEX": request.form["SEX"],
            "EDUCATION": request.form["EDUCATION"],
            "MARRIAGE": request.form["MARRIAGE"],

            "PAY_1": request.form["PAY_1"],
            "PAY_2": request.form["PAY_2"],
            "PAY_3": request.form["PAY_3"],
            "PAY_4": request.form["PAY_4"],
            "PAY_5": request.form["PAY_5"],
            "PAY_6": request.form["PAY_6"],

            "BILL_AMT1": float(request.form["BILL_AMT1"]),
            "BILL_AMT2": float(request.form["BILL_AMT2"]),
            "BILL_AMT3": float(request.form["BILL_AMT3"]),
            "BILL_AMT4": float(request.form["BILL_AMT4"]),
            "BILL_AMT5": float(request.form["BILL_AMT5"]),
            "BILL_AMT6": float(request.form["BILL_AMT6"]),

            "PAY_AMT1": float(request.form["PAY_AMT1"]),
            "PAY_AMT2": float(request.form["PAY_AMT2"]),
            "PAY_AMT3": float(request.form["PAY_AMT3"]),
            "PAY_AMT4": float(request.form["PAY_AMT4"]),
            "PAY_AMT5": float(request.form["PAY_AMT5"]),
            "PAY_AMT6": float(request.form["PAY_AMT6"]),
        }

        df = pd.DataFrame([data])

        pipeline = PredictPipeline()
        prob = pipeline.predict(df)[0]

        result = {
            "probability": round(float(prob), 4),
            "risk": "High" if prob >= 0.5 else "Low"
        }

    return render_template("index.html", result=result)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True, use_reloader=False)
