import os
import gc
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Tắt log thừa
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# --- CHIÊU CUỐI: CƯỠNG CHẾ ĐỔI TÊN BIẾN ĐỂ KHỚP KERAS 2 ---
from tensorflow.keras.layers import InputLayer

class PatchedInputLayer(InputLayer):
    def __init__(self, *args, **kwargs):
        # Đổi batch_shape (Keras 3) thành batch_input_shape (Keras 2)
        if 'batch_shape' in kwargs:
            kwargs['batch_input_shape'] = kwargs.pop('batch_shape')
        kwargs.pop('optional', None)
        super().__init__(*args, **kwargs)

    @classmethod
    def from_config(cls, config):
        if 'batch_shape' in config:
            config['batch_input_shape'] = config.pop('batch_shape')
        config.pop('optional', None)
        return super(PatchedInputLayer, cls).from_config(config)

# Đăng ký đè lớp InputLayer hệ thống bằng lớp Patched của mình
tf.keras.utils.get_custom_objects().update({'InputLayer': PatchedInputLayer})
# --------------------------------------------------------

app = FastAPI()
model = None

try:
    # Thử nạp .keras trước, sau đó là .h5
    for m_file in ["depression_lstm_model.keras", "depression_lstm_model.h5"]:
        if os.path.exists(m_file):
            model = tf.keras.models.load_model(
                m_file, 
                custom_objects={'InputLayer': PatchedInputLayer}, 
                compile=False
            )
            print(f"[OK] Da nap model {m_file} thanh cong!")
            break
    
    if model is None:
        print("[x] Khong tim thay file model nao!")
except Exception as e:
    print(f"[x] Loi nap model: {e}")

class FeatureData(BaseModel):
    features: list

def preprocess_input(features, max_len=300, feat_dim=161):
    matrix = np.array(features, dtype=np.float32)
    curr_len = matrix.shape[0]
    if curr_len > max_len:
        matrix = matrix[:max_len, :]
    elif curr_len < max_len:
        padding = np.zeros((max_len - curr_len, feat_dim), dtype=np.float32)
        matrix = np.vstack((matrix, padding))
    return np.expand_dims(matrix, axis=0)

@app.get("/")
async def root():
    return {"message": "Server dang chay!"}

@app.post("/predict")
async def predict_depression(data: FeatureData):
    if model is None:
        raise HTTPException(status_code=500, detail="Model chua duoc nap!")
    try:
        input_data = preprocess_input(data.features)
        prediction = model.predict(input_data, verbose=0)
        probability = float(prediction[0][0])
        status = "Tram cam" if probability > 0.5 else "Binh thuong"
        return {
            "status": "success",
            "prediction": status,
            "confidence": f"{round(probability * 100, 2)}%",
            "advice": "Hay nghi ngoi nhe!" if status == "Tram cam" else "Trang thai tot!"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)