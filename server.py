import os
import gc
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Tắt log thừa
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# --- BỘ LỌC TỪ KHÓA RÁC CHO LỚP DENSE ---
from tensorflow.keras.layers import Dense

class PatchedDense(Dense):
    def __init__(self, *args, **kwargs):
        # Cắt bỏ cái đuôi quantization_config vô dụng
        kwargs.pop('quantization_config', None)
        super().__init__(*args, **kwargs)

    @classmethod
    def from_config(cls, config):
        config.pop('quantization_config', None)
        return super().from_config(config)

# Đăng ký lớp Dense đã được "lọc" vào hệ thống
tf.keras.utils.get_custom_objects().update({'Dense': PatchedDense})
# ----------------------------------------

app = FastAPI()
model = None

# Cấu hình tiết kiệm RAM
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

try:
    model_path = "depression_lstm_model.keras" 
    if not os.path.exists(model_path):
        model_path = "depression_lstm_model.h5"

    if os.path.exists(model_path):
        # Ép nó dùng lớp Dense đã vá của mình
        model = tf.keras.models.load_model(
            model_path, 
            custom_objects={'Dense': PatchedDense},
            compile=False
        )
        print(f"[OK] Da nap model {model_path} thanh cong!")
        gc.collect()
    else:
        print("[x] Khong tim thay file model nao ca!")
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
    return {"message": "Server AI dang chay ruc ro!"}

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
            "advice": "Hay danh thoi gian nghi ngoi nhe!" if status == "Tram cam" else "Trang thai cua ban rat tot!"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)