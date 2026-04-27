import os
import gc

# Tắt log thừa
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np

# Gọi TensorFlow chính hãng ra
import tensorflow as tf

# Ép chạy đơn luồng tiết kiệm RAM tối đa
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

app = FastAPI()

# Biến toàn cục
model = None

try:
    model_path = "depression_lstm_model.h5"
    if os.path.exists(model_path):
        # Nạp mô hình Keras H5 trực tiếp
        model = tf.keras.models.load_model(model_path)
        print("[OK] Keras H5 model loaded successfully!")
        gc.collect() # Dọn rác ngay lập tức
    else:
        print(f"[x] File not found: {model_path}")
except Exception as e:
    print(f"[x] Error loading model: {e}")

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
    return {"message": "Server AI (TF-CPU + H5 Mode) dang chay!"}

@app.post("/predict")
async def predict_depression(data: FeatureData):
    if model is None:
        raise HTTPException(status_code=500, detail="Model chua duoc nap!")
    
    try:
        input_data = preprocess_input(data.features)
        
        # Dự đoán trực tiếp bằng model Keras
        prediction = model.predict(input_data, verbose=0)
        probability = float(prediction[0][0])
        
        status = "Tram cam" if probability > 0.5 else "Binh thuong"
        risk_percent = f"{round(probability * 100, 2)}%"
        
        gc.collect() # Dọn rác sau khi dự đoán
        
        return {
            "status": "success",
            "prediction": status,
            "confidence": risk_percent,
            "advice": "Hay danh thoi gian nghi ngoi nhieu hon nhe!" if status == "Tram cam" else "Trang thai cua ban rat tot!"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)