from flask import Flask, render_template, request
import os
from detection import detect_vehicles

app = Flask(__name__)

@app.route("/", methods=["GET","POST"])
def index():
    if request.method == "POST":
        file = request.files["video"]

        input_path = os.path.join("static/uploads", file.filename)
        output_path = os.path.join("static/outputs", "output.mp4")

        file.save(input_path)

        detect_vehicles(input_path, output_path)

        return render_template("index.html", video="outputs/output.mp4")

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)