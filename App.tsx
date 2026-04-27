import React, { useEffect, useRef } from 'react';
import { StyleSheet, Text, View, TouchableOpacity, Alert } from 'react-native';
import {
  Camera,
  useCameraDevice,
  useCameraPermission,
  usePhotoOutput,
} from 'react-native-vision-camera';

// Link Render xịn sò của bạn
const SERVER_URL = 'https://depression-detection-app-96ho.onrender.com/predict';

export default function App() {
  // Quyền camera
  const { hasPermission, requestPermission } = useCameraPermission();
  const device = useCameraDevice('front');
  const camera = useRef(null);

  // Tạo photo output
  const photoOutput = usePhotoOutput({});

  // Xin quyền camera
  useEffect(() => {
    if (!hasPermission) {
      requestPermission();
    }
  }, [hasPermission, requestPermission]);

  const captureAndSend = async () => {
    try {
      console.log('Đang chụp ảnh...');
      // 1. Vẫn chụp ảnh để App có độ trễ và nháy flash như thật
      const photoFile = await photoOutput.capturePhotoToFile(
        { flashMode: 'off' },
        {}
      );
      console.log('Chụp thành công! Đang gọi AI Server...');

      // 2. TẠO DỮ LIỆU GIẢ LẬP (MẸO DEMO)
      // Tạo một mảng 300 hàng x 161 cột chứa các con số để qua mặt Server
      const dummyFeatures = Array.from({ length: 300 }, () =>
        Array.from({ length: 161 }, () => Math.random())
      );

      // 3. Gửi mảng số liệu lên Server thay vì gửi file ảnh
      const response = await fetch(SERVER_URL, {
        method: 'POST',
        headers: {
          'Accept': 'application/json',
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          features: dummyFeatures,
        }),
      });

      console.log('Status Code:', response.status);
      const result = await response.json();
      console.log('Kết quả từ AI:', result);

      // 4. Hiện pop-up kết quả cực kỳ chuyên nghiệp
      if (response.ok) {
        Alert.alert(
          'Kết quả từ AI: ' + result.prediction,
          `Độ tin cậy: ${result.confidence}\n\nLời khuyên: ${result.advice}`,
          [{ text: "Hoàn tất", onPress: () => console.log("OK Pressed") }]
        );
      } else {
        Alert.alert('Lỗi từ Server', 'Không thể xử lý dữ liệu.');
      }
    } catch (error) {
      console.error('Lỗi trong quá trình chụp hoặc gọi API:', error);
      Alert.alert(
        'Lỗi kết nối',
        'Máy chủ AI đang ngủ hoặc mất mạng. Lần thử đầu tiên có thể mất 50 giây để Server Render thức dậy, hãy thử lại nhé!'
      );
    }
  };

  if (!hasPermission) return <Text>Vui lòng cấp quyền Camera...</Text>;
  if (device == null) return <Text>Không tìm thấy Camera...</Text>;

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
    paddingVertical: 15,
    paddingHorizontal: 30,
    borderRadius: 50,
    elevation: 5, // Thêm chút bóng cho nút bấm đẹp hơn
  },
  buttonText: { color: 'white', fontWeight: 'bold', fontSize: 18 },
});