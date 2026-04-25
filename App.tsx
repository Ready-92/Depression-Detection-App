import React, { useEffect, useRef } from 'react';
import { StyleSheet, Text, View, TouchableOpacity, Alert } from 'react-native';
import {
  Camera,
  useCameraDevice,
  useCameraPermission,
  usePhotoOutput,
  type CameraRef,
} from 'react-native-vision-camera';

export default function App() {
  // Quyền camera (API v5)
  const { hasPermission, requestPermission } = useCameraPermission();
  const device = useCameraDevice('front'); // Dùng camera trước
  const camera = useRef<CameraRef>(null);

  // Tạo photo output (API v5)
  const photoOutput = usePhotoOutput({});

  // Xin quyền camera khi vừa mở App
  useEffect(() => {
    if (!hasPermission) {
      requestPermission();
    }
  }, [hasPermission, requestPermission]);

  // Hàm chụp ảnh và gửi lên Backend (API v5)
  const captureAndSend = async () => {
    try {
      // 1. Chụp ảnh bằng photoOutput
      const photoFile = await photoOutput.capturePhotoToFile(
        { flashMode: 'off' },
        {},
      );
      console.log('Đã chụp ảnh lưu tại:', photoFile.filePath);

      // 2. Tạm thời gửi dữ liệu số (Mock Data) lên để test AI Server trước
      console.log('Đang gọi AI Server...');
      const mockFrame = Array.from({ length: 161 }, () => Math.random());

      // QUAN TRỌNG: Sửa dòng này thành IP WiFi của máy tính bạn!
      const SERVER_URL = 'http://10.0.2.2:8000/predict';

      const response = await fetch(SERVER_URL, {
        method: 'POST',
        headers: {
          Accept: 'application/json',
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ features: [mockFrame] }),
      });

      const result = await response.json();

      // 3. Hiện kết quả ra màn hình App
      Alert.alert(
        'Kết quả từ AI: ' + result.prediction,
        `Độ tin cậy: ${result.confidence}\n\nLời khuyên: ${result.advice}`,
      );
    } catch (error) {
      console.error('Lỗi trong quá trình chụp hoặc gọi API:', error);
      Alert.alert(
        'Lỗi kết nối',
        'Không thể gọi tới Server. Hãy kiểm tra lại IP.',
      );
    }
  };

  if (!hasPermission) return <Text>Vui lòng cấp quyền Camera</Text>;
  if (device == null) return <Text>Không tìm thấy Camera</Text>;

  return (
    <View style={styles.container}>
      <Camera
        ref={camera}
        style={StyleSheet.absoluteFill}
        device={device}
        isActive={true}
        outputs={[photoOutput]}
      />
      <TouchableOpacity style={styles.button} onPress={captureAndSend}>
        <Text style={styles.buttonText}>Quét Cảm Xúc</Text>
      </TouchableOpacity>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, justifyContent: 'flex-end', alignItems: 'center' },
  button: {
    marginBottom: 50,
    backgroundColor: '#ff4757',
    padding: 15,
    borderRadius: 50,
  },
  buttonText: { color: 'white', fontWeight: 'bold', fontSize: 16 },
});