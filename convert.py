import tensorflow as tf

# 1. Đọc file h5 cũ (bỏ qua lỗi compile)
model = tf.keras.models.load_model("depression_lstm_model.h5", compile=False)

# 2. Lưu lại bằng định dạng .keras mới nhất
model.save("depression_lstm_model.keras")

print("Chuyen doi thanh cong sang .keras! Tuyet voi!")