import os
import gc
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Tắt log thừa
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

app = FastAPI()
model = None

# Cấu hình tiết kiệm RAM cho Server
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

# 1. TỰ TẠO LẠI KHUNG MÔ HÌNH (Dựa 100% vào log của Việt)
def build_model():
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Input, Masking, LSTM, Dropout, Dense
    
    m = Sequential()
    m.add(Input(shape=(300, 161)))
    m.add(Masking(mask_value=0.0))
    m.add(LSTM(128, return_sequences=True))
    m.add(Dropout(0.3))
    m.add(LSTM(64, return_sequences=False))
    m.add(Dropout(0.3))
    m.add(Dense(32, activation='relu'))
    m.add(Dense(1, activation='sigmoid'))
    return m

try:
    model_path = "depression_lstm_model.keras" 
    if not os.path.exists(model_path):
        model_path = "depression_lstm_model.h5"

    if os.path.exists(model_path):
        print("[INFO] Dang tao kien truc mo hinh...")
        model = build_model()
        
        print(f"[INFO] Dang nap TRONG SO (weights) tu {model_path}...")
        # 2. CHỈ NẠP TRỌNG SỐ (WEIGHTS) - BỎ QUA HOÀN TOÀN KHÂU ĐỌC CONFIG LỖI
        model.load_weights(model_path)
        
        print(f"[OK] Da nap model thanh cong ruc ro!")
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
        status = "Trầm cảm" if probability > 0.5 else "Bình thường"
        return {
            "status": "success",
            "prediction": status,
            "confidence": f"{round(probability * 100, 2)}%",
            "advice": "Hãy dành thời gian để nghỉ ngơi nhé!" if status == "Trầm cảm" else "Trạng thái của bạn rấy tốt!"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)