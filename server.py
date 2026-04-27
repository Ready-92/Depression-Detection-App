import os
# 1. Tối ưu hệ thống TRƯỚC KHI import tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Chỉ hiện lỗi, tắt các log cảnh báo nặng nề
os.environ['OTTO_MAX_THREADS'] = '1'      # Giới hạn thread để tiết kiệm RAM

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np

# Sử dụng tensorflow.lite nếu có thể, nhưng ở đây mình tối ưu tensorflow thường
import tensorflow as tf

app = FastAPI()

# 2. Cấu hình TensorFlow tiết kiệm tài nguyên
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

# Biến toàn cục để giữ mô hình
model = None

# Nạp mô hình một cách cẩn thận
try:
    model_path = "depression_lstm_model.h5"
    if os.path.exists(model_path):
        # compile=False giúp nạp nhanh hơn và tốn ít RAM hơn vì không cần nạp optimizer
        model = tf.keras.models.load_model(model_path, compile=False)
        print("[✓] Đã nạp mô hình AI thành công!")
    else:
        print(f"[x] Không tìm thấy file: {model_path}")
except Exception as e:
    print(f"[x] Lỗi nạp mô hình: {e}")

class FeatureData(BaseModel):
    features: list 

def preprocess_input(features, max_len=300, feat_dim=161):
    matrix = np.array(features)
    curr_len = matrix.shape[0]
    
    if curr_len > max_len:
        matrix = matrix[:max_len, :]
    elif curr_len < max_len:
        padding = np.zeros((max_len - curr_len, feat_dim))
        matrix = np.vstack((matrix, padding))
    
    return np.expand_dims(matrix, axis=0).astype(np.float32) # Ép kiểu float32 cho nhẹ

@app.get("/")
async def root():
    return {"message": "Server AI đang chạy (Render 512MB Mode)"}

@app.post("/predict")
async def predict_depression(data: FeatureData):
    if model is None:
        raise HTTPException(status_code=500, detail="Model chưa được nạp!")
    
    try:
        input_data = preprocess_input(data.features)
        
        # Dự đoán
        prediction = model.predict(input_data, verbose=0) # verbose=0 để không in log khi chạy
        probability = float(prediction[0][0])
        
        status = "Trầm cảm" if probability > 0.5 else "Bình thường"
        risk_percent = f"{round(probability * 100, 2)}%"
        
        return {
            "status": "success",
            "prediction": status,
            "confidence": risk_percent,
            "advice": "Hãy dành thời gian nghỉ ngơi nhiều hơn nhé!" if status == "Trầm cảm" else "Trạng thái của bạn rất tốt!"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # Render sẽ dùng biến môi trường PORT, nếu không có thì mặc định 8000
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)