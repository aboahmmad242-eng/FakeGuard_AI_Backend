
from flask_cors import CORS
from flask import Flask, request, jsonify
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


app = Flask(__name__)
CORS(app)


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


model_path = "eleen1999/FakeGuard-AI-BERT-FakeNews-Detector"


tokenizer = AutoTokenizer.from_pretrained(
    model_path
)


model = AutoModelForSequenceClassification.from_pretrained(
    model_path
)

model.to(device)

model.eval()


label_mapping = {
    0: "Real News",
    1: "Fake News"
}



@app.route("/", methods=["GET"])
def home():

    return jsonify(
        {
            "message":
            "Fake News Detection API is running"
        }
    )



@app.route("/predict", methods=["POST"])
def predict():


    data = request.json


    article_text = data.get(
        "text",
        ""
    )


    if article_text == "":

        return jsonify(
            {
                "error":
                "No text provided"
            }
        )



    inputs = tokenizer(

        article_text,

        padding=True,

        truncation=True,

        max_length=256,

        return_tensors="pt"

    )


    inputs = {

        key:value.to(device)

        for key,value in inputs.items()

    }



    with torch.no_grad():

        outputs = model(
            **inputs
        )


    probabilities = torch.softmax(
        outputs.logits,
        dim=1
    )


    prediction_id = torch.argmax(
        probabilities,
        dim=1
    ).item()



    response = {


        "prediction":
        label_mapping[prediction_id],


        "confidence":
        float(
            probabilities[0][prediction_id]
        ),


        "real_news_probability":
        float(
            probabilities[0][0]
        ),


        "fake_news_probability":
        float(
            probabilities[0][1]
        )

    }


    return jsonify(response)



if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000
    )
