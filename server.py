from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

app = FastAPI()

# 1. Nạp bộ não AI đã huấn luyện
# Đảm bảo file .h5 nằm cùng thư mục với server.py
try:
    model = load_model("depression_lstm_model.h5")
    print("[\u2713] Đã nạp mô hình AI thành công!")
except Exception as e:
    print(f"[x] Lỗi nạp mô hình: {e}")

class FeatureData(BaseModel):
    # Điện thoại sẽ gửi lên một mảng các đặc trưng (features)
    # data: List[List[float]]
    features: list 

def preprocess_input(features, max_len=300, feat_dim=161):
    """Hàm chuẩn hóa dữ liệu gửi từ điện thoại về dạng AI hiểu được"""
    matrix = np.array(features)
    curr_len = matrix.shape[0]
    
    if curr_len > max_len:
        matrix = matrix[:max_len, :]
    elif curr_len < max_len:
        padding = np.zeros((max_len - curr_len, feat_dim))
        matrix = np.vstack((matrix, padding))
    
    # Biến đổi về dạng (1, 300, 161) để đưa vào mạng LSTM
    return np.expand_dims(matrix, axis=0)

@app.post("/predict")
async def predict_depression(data: FeatureData):
    try:
        # 2. Xử lý dữ liệu đầu vào
        input_data = preprocess_input(data.features)
        
        # 3. Cho AI dự đoán
        prediction = model.predict(input_data)
        probability = float(prediction[0][0])
        
        # 4. Phân loại kết quả
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
    # Sử dụng host 0.0.0.0 để điện thoại có thể kết nối vào qua mạng WiFi
    uvicorn.run(app, host="0.0.0.0", port=8000)