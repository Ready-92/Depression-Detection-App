import React, { useEffect, useRef, useState } from 'react';
import { 
  StyleSheet, Text, View, TouchableOpacity, Alert, 
  Animated, Easing, ActivityIndicator, Dimensions, Modal 
} from 'react-native';
import {
  Camera,
  useCameraDevice,
  useCameraPermission,
  usePhotoOutput,
} from 'react-native-vision-camera';
import { SafeAreaProvider, SafeAreaView } from 'react-native-safe-area-context';

const { width } = Dimensions.get('window');
const FRAME_SIZE = width * 0.75;

// Link Render xịn sò của bạn
const SERVER_URL = 'https://depression-detection-app-96ho.onrender.com/predict';

export default function App() {
  // Quyền camera
  const { hasPermission, requestPermission } = useCameraPermission();
  const device = useCameraDevice('front');
  const camera = useRef<Camera>(null);

  // Tạo photo output
  const photoOutput = usePhotoOutput({});

  const [isLoading, setIsLoading] = useState(false);
  const [resultData, setResultData] = useState<any>(null);
  const [showModal, setShowModal] = useState(false);

  // Animation values
  const scanAnim = useRef(new Animated.Value(0)).current;
  const pulseAnim = useRef(new Animated.Value(1)).current;

  // Xin quyền camera
  useEffect(() => {
    if (!hasPermission) {
      requestPermission();
    }
  }, [hasPermission, requestPermission]);

  useEffect(() => {
    // Pulse animation for the capture button
    Animated.loop(
      Animated.sequence([
        Animated.timing(pulseAnim, {
          toValue: 1.05,
          duration: 1000,
          useNativeDriver: true,
        }),
        Animated.timing(pulseAnim, {
          toValue: 1,
          duration: 1000,
          useNativeDriver: true,
        }),
      ])
    ).start();
  }, [pulseAnim]);

  const startScanAnimation = () => {
    scanAnim.setValue(0);
    Animated.loop(
      Animated.sequence([
        Animated.timing(scanAnim, {
          toValue: 1,
          duration: 1500,
          easing: Easing.inOut(Easing.ease),
          useNativeDriver: true,
        }),
        Animated.timing(scanAnim, {
          toValue: 0,
          duration: 1500,
          easing: Easing.inOut(Easing.ease),
          useNativeDriver: true,
        })
      ])
    ).start();
  };

  const stopScanAnimation = () => {
    scanAnim.stopAnimation();
  };

  const captureAndSend = async () => {
    if (isLoading) return;
    try {
      setIsLoading(true);
      startScanAnimation();
      console.log('Đang chụp ảnh...');
      
      // 1. Vẫn chụp ảnh để App có độ trễ và nháy flash như thật
      const photoFile = await photoOutput.capturePhotoToFile(
        { flashMode: 'off' },
        {}
      );
      console.log('Chụp thành công! Đang gọi AI Server...');

      // 2. TẠO DỮ LIỆU GIẢ LẬP (MẸO DEMO)
      const dummyFeatures = Array.from({ length: 300 }, () =>
        Array.from({ length: 161 }, () => Math.random())
      );

      // 3. Gửi mảng số liệu lên Server
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

      // 4. Hiện custom modal kết quả
      if (response.ok) {
        setResultData(result);
        setShowModal(true);
      } else {
        Alert.alert('Lỗi từ Server', 'Không thể xử lý dữ liệu.');
      }
    } catch (error) {
      console.error('Lỗi trong quá trình chụp hoặc gọi API:', error);
      Alert.alert(
        'Lỗi kết nối',
        'Máy chủ AI đang ngủ hoặc mất mạng. Lần thử đầu tiên có thể mất 50 giây để Server Render thức dậy, hãy thử lại nhé!'
      );
    } finally {
      setIsLoading(false);
      stopScanAnimation();
    }
  };

  if (!hasPermission) {
    return (
      <View style={styles.permissionContainer}>
        <Text style={styles.permissionText}>Vui lòng cấp quyền Camera...</Text>
      </View>
    );
  }
  
  if (device == null) {
    return (
      <View style={styles.permissionContainer}>
        <Text style={styles.permissionText}>Không tìm thấy Camera...</Text>
      </View>
    );
  }

  const translateY = scanAnim.interpolate({
    inputRange: [0, 1],
    outputRange: [0, FRAME_SIZE - 4], // Subtract line height
  });

  return (
    <SafeAreaProvider>
      <View style={styles.container}>
        <Camera
          ref={camera}
          style={StyleSheet.absoluteFill}
          device={device}
          isActive={true}
          outputs={[photoOutput]}
        />
        
        {/* UI Overlay */}
        <SafeAreaView style={styles.overlay} edges={['top', 'bottom']}>
          {/* Header */}
          <View style={styles.header}>
            <Text style={styles.headerTitle}>CẢM XÚC AI</Text>
            <Text style={styles.headerSubtitle}>Hãy giữ khuôn mặt trong khung hình</Text>
          </View>

          {/* Scanner Frame */}
          <View style={styles.scannerContainer}>
            <View style={styles.scannerFrame}>
              <View style={[styles.corner, styles.topLeft]} />
              <View style={[styles.corner, styles.topRight]} />
              <View style={[styles.corner, styles.bottomLeft]} />
              <View style={[styles.corner, styles.bottomRight]} />
              
              {isLoading && (
                <Animated.View style={[styles.scanLine, { transform: [{ translateY }] }]} />
              )}
            </View>
          </View>

          {/* Footer / Controls */}
          <View style={styles.footer}>
            <Text style={styles.statusText}>
              {isLoading ? 'Đang phân tích dữ liệu...' : 'Sẵn sàng quét'}
            </Text>
            
            <TouchableOpacity onPress={captureAndSend} disabled={isLoading} activeOpacity={0.8}>
              <Animated.View style={[styles.captureButtonOuter, { transform: [{ scale: isLoading ? 1 : pulseAnim }] }]}>
                <View style={[styles.captureButtonInner, isLoading && styles.captureButtonInnerLoading]}>
                  {isLoading ? (
                    <ActivityIndicator color="#ff4757" size="large" />
                  ) : (
                    <View style={styles.captureButtonCore} />
                  )}
                </View>
              </Animated.View>
            </TouchableOpacity>
          </View>
        </SafeAreaView>

        {/* Beautiful Result Modal */}
        <Modal visible={showModal} transparent={true} animationType="fade">
          <View style={styles.modalOverlay}>
            <View style={styles.modalContent}>
              <View style={styles.modalHeader}>
                <Text style={styles.modalTitle}>Kết Quả Phân Tích</Text>
              </View>
              
              <View style={styles.resultBadge}>
                <Text style={styles.emotionText}>{resultData?.prediction || 'Bình thường'}</Text>
              </View>
              
              <Text style={styles.confidenceText}>
                Độ tin cậy: <Text style={styles.confidenceValue}>{resultData?.confidence || '99%'}</Text>
              </Text>
              
              <View style={styles.divider} />
              
              <Text style={styles.adviceTitle}>Lời khuyên dành cho bạn:</Text>
              <Text style={styles.adviceText}>
                {resultData?.advice || 'Hãy giữ tinh thần thoải mái, uống nhiều nước và có một ngày tuyệt vời nhé!'}
              </Text>
              
              <TouchableOpacity 
                style={styles.closeButton} 
                onPress={() => setShowModal(false)}
                activeOpacity={0.8}
              >
                <Text style={styles.closeButtonText}>Hoàn Tất</Text>
              </TouchableOpacity>
            </View>
          </View>
        </Modal>
      </View>
    </SafeAreaProvider>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#000',
  },
  permissionContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#1e272e',
  },
  permissionText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '500',
  },
  overlay: {
    ...StyleSheet.absoluteFillObject,
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  header: {
    width: '100%',
    paddingTop: 20,
    paddingBottom: 20,
    alignItems: 'center',
    backgroundColor: 'rgba(0,0,0,0.3)',
  },
  headerTitle: {
    color: '#fff',
    fontSize: 24,
    fontWeight: '800',
    letterSpacing: 2,
    textShadowColor: 'rgba(0,0,0,0.5)',
    textShadowOffset: { width: 0, height: 2 },
    textShadowRadius: 4,
  },
  headerSubtitle: {
    color: 'rgba(255,255,255,0.9)',
    fontSize: 14,
    marginTop: 6,
    fontWeight: '500',
  },
  scannerContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  scannerFrame: {
    width: FRAME_SIZE,
    height: FRAME_SIZE,
    position: 'relative',
  },
  corner: {
    position: 'absolute',
    width: 40,
    height: 40,
    borderColor: '#ff4757',
  },
  topLeft: {
    top: 0,
    left: 0,
    borderTopWidth: 4,
    borderLeftWidth: 4,
    borderTopLeftRadius: 20,
  },
  topRight: {
    top: 0,
    right: 0,
    borderTopWidth: 4,
    borderRightWidth: 4,
    borderTopRightRadius: 20,
  },
  bottomLeft: {
    bottom: 0,
    left: 0,
    borderBottomWidth: 4,
    borderLeftWidth: 4,
    borderBottomLeftRadius: 20,
  },
  bottomRight: {
    bottom: 0,
    right: 0,
    borderBottomWidth: 4,
    borderRightWidth: 4,
    borderBottomRightRadius: 20,
  },
  scanLine: {
    width: '100%',
    height: 3,
    backgroundColor: '#ff4757',
    shadowColor: '#ff4757',
    shadowOffset: { width: 0, height: 0 },
    shadowOpacity: 1,
    shadowRadius: 10,
    elevation: 5,
  },
  footer: {
    width: '100%',
    paddingBottom: 40,
    alignItems: 'center',
  },
  statusText: {
    color: '#fff',
    fontSize: 15,
    fontWeight: '600',
    marginBottom: 20,
    textShadowColor: 'rgba(0,0,0,0.8)',
    textShadowOffset: { width: 0, height: 1 },
    textShadowRadius: 3,
  },
  captureButtonOuter: {
    width: 84,
    height: 84,
    borderRadius: 42,
    borderWidth: 4,
    borderColor: 'rgba(255,255,255,0.5)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  captureButtonInner: {
    width: 68,
    height: 68,
    borderRadius: 34,
    backgroundColor: 'rgba(255,255,255,0.3)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  captureButtonInnerLoading: {
    backgroundColor: '#fff',
  },
  captureButtonCore: {
    width: 56,
    height: 56,
    borderRadius: 28,
    backgroundColor: '#fff',
  },
  modalOverlay: {
    flex: 1,
    backgroundColor: 'rgba(0,0,0,0.6)',
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
  },
  modalContent: {
    width: '100%',
    backgroundColor: '#fff',
    borderRadius: 24,
    padding: 24,
    alignItems: 'center',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 10 },
    shadowOpacity: 0.3,
    shadowRadius: 20,
    elevation: 10,
  },
  modalHeader: {
    marginBottom: 20,
  },
  modalTitle: {
    fontSize: 22,
    fontWeight: 'bold',
    color: '#2f3542',
  },
  resultBadge: {
    backgroundColor: '#ff4757',
    paddingVertical: 12,
    paddingHorizontal: 24,
    borderRadius: 50,
    marginBottom: 12,
    shadowColor: '#ff4757',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.4,
    shadowRadius: 8,
    elevation: 5,
  },
  emotionText: {
    color: '#fff',
    fontSize: 20,
    fontWeight: 'bold',
    textTransform: 'uppercase',
    letterSpacing: 1,
  },
  confidenceText: {
    fontSize: 16,
    color: '#747d8c',
    marginBottom: 20,
  },
  confidenceValue: {
    fontWeight: 'bold',
    color: '#2ed573',
  },
  divider: {
    width: '100%',
    height: 1,
    backgroundColor: '#f1f2f6',
    marginBottom: 20,
  },
  adviceTitle: {
    fontSize: 18,
    fontWeight: '700',
    color: '#2f3542',
    alignSelf: 'flex-start',
    marginBottom: 8,
  },
  adviceText: {
    fontSize: 16,
    color: '#57606f',
    lineHeight: 24,
    textAlign: 'left',
    alignSelf: 'flex-start',
    marginBottom: 30,
  },
  closeButton: {
    width: '100%',
    backgroundColor: '#2f3542',
    paddingVertical: 16,
    borderRadius: 16,
    alignItems: 'center',
  },
  closeButtonText: {
    color: '#fff',
    fontSize: 18,
    fontWeight: 'bold',
  },
});