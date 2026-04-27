import React, { useEffect, useRef } from 'react';
import { StyleSheet, Text, View, TouchableOpacity, Alert } from 'react-native';
import {
  Camera,
  useCameraDevice,
  useCameraPermission,
  usePhotoOutput,
  type CameraRef,
} from 'react-native-vision-camera';

// Thay link này bằng link "Forwarding" mà Ngrok cấp cho bạn khi chạy lệnh
// Ví dụ: https://abcd-123-456.ngrok-free.app
const NGROK_URL = 'https://dipteral-eleanor-ungrainable.ngrok-free.dev';
const SERVER_URL = `${NGROK_URL}/predict`;

export default function App() {
  // Quyền camera
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

  // Hàm chụp ảnh và gửi lên Backend
  const captureAndSend = async () => {
    try {
      // 1. Chụp ảnh ra file
      const photoFile = await photoOutput.capturePhotoToFile(
        { flashMode: 'off' },
        {},
      );
      console.log('Đã chụp ảnh lưu tại:', photoFile.filePath);

      // 2. Tạo FormData để gửi ảnh lên Server
      const formData = new FormData();
      formData.append('file', {
        uri: `file://${photoFile.filePath}`,
        type: 'image/jpeg',
        name: 'photo.jpg',
      } as any);

      // 3. Gọi API Server
      console.log('Đang gọi AI Server...');
      const response = await fetch(SERVER_URL, {
        method: 'POST',
        body: formData,
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      console.log('Status Code:', response.status);
      const result = await response.json();

      // 4. Hiện kết quả ra màn hình App
      Alert.alert(
        'Kết quả từ AI: ' + result.prediction,
        `Độ tin cậy: ${result.confidence}\n\nLời khuyên: ${result.advice}`,
      );
    } catch (error) {
      console.error('Lỗi trong quá trình chụp hoặc gọi API:', error);
      Alert.alert(
        'Lỗi kết nối',
        'Không thể gọi tới Server. Hãy kiểm tra lại Ngrok đã bật chưa và link đã đúng chưa.',
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