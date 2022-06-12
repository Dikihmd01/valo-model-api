import tensorflow as tf
import numpy as np

from fastapi import FastAPI

app = FastAPI()

# @app.get('/recommendations/{vac}')
# async def recommendations(vac):
#     return vac

@app.get("/api/recommendations/{vac}")
async def home(vac):
    labels = ["AZ", "Sinovac", "Sinopharm", "Pfizer", "Moderna", "Janssen"]
    vac = vac.lower()
    label_lower = [label.lower() for label in labels]

    if vac.isalpha() and vac in label_lower:
        interpreter = tf.lite.Interpreter(model_path="converted_model.tflite")
        interpreter.allocate_tensors()

        # get input and output tensors
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        interpreter.set_tensor(input_details[0]['index'], np.array([vac]))

        interpreter.invoke()

        output_data = interpreter.get_tensor(output_details[1]['index'])
        vaccines = [x.decode() for x in output_data[0]]

        dataJsonified = {
            "recommendations": vaccines,
            "message": "Data berhasil didapatkan"
        }
        return dataJsonified
    else:
        return {"message": "Masukkan data dengan benar"}

