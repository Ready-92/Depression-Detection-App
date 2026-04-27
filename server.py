import os
import gc
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np

# CHỈ DÙNG TFLITE-RUNTIME SIÊU NHẸ (5MB)
import tflite_runtime.interpreter as tflite

app = FastAPI()

# Biến toàn cục
interpreter = None

try:
    model_path = "depression_model.tflite"
    if os.path.exists(model_path):
        # Nạp mô hình bằng tflite-runtime
        interpreter = tflite.Interpreter(model_path=model_path, num_threads=1)
        interpreter.allocate_tensors()
        
        # Lấy thông tin đầu vào/đầu ra ngay từ đầu để tiết kiệm thời gian
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print("[OK] TFLite Pure model loaded!")
        gc.collect() # Dọn rác
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
    return {"message": "Server AI dang chay (Render 512MB - Pure TFLite Mode)"}

@app.post("/predict")
async def predict_depression(data: FeatureData):
    if interpreter is None:
        raise HTTPException(status_code=500, detail="Model chua duoc nap!")
    
    try:
        input_data = preprocess_input(data.features)
        
        # Đưa dữ liệu vào dự đoán
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        
        # Lấy kết quả
        prediction = interpreter.get_tensor(output_details[0]['index'])
        probability = float(prediction[0][0])
        
        status = "Tram cam" if probability > 0.5 else "Binh thuong"
        risk_percent = f"{round(probability * 100, 2)}%"
        
        gc.collect()
        
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