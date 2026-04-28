import os
import gc
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Tắt log thừa để Server chạy nhẹ hơn
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# --- ĐÂY LÀ "BÙA" VÁ LỖI INPUTLAYER CHO KERAS 3 ---
from tensorflow.keras.layers import InputLayer

class PatchedInputLayer(InputLayer):
    """Tự động loại bỏ các tham số 'batch_shape' và 'optional' gây lỗi trên Keras 3"""
    def __init__(self, *args, **kwargs):
        kwargs.pop('batch_shape', None)
        kwargs.pop('optional', None)
        super().__init__(*args, **kwargs)

# -----------------------------------------------

app = FastAPI()
model = None

# Cấu hình tiết kiệm tài nguyên cho gói Free của Render
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

try:
    # Ưu tiên nạp file .keras trước, nếu không có thì tìm file .h5
    model_path = "depression_lstm_model.keras" 
    if not os.path.exists(model_path):
        model_path = "depression_lstm_model.h5"

    if os.path.exists(model_path):
        # Ép Keras dùng PatchedInputLayer để nạp model cũ vào môi trường mới
        model = tf.keras.models.load_model(
            model_path, 
            custom_objects={'InputLayer': PatchedInputLayer}, 
            compile=False
        )
        print(f"[OK] Da nap model {model_path} thanh cong!")
        gc.collect()
    else:
        print("[x] Khong tim thay file model nao trong thu muc!")
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