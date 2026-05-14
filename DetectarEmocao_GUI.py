import os

import cv2
import numpy as np
import onnxruntime
from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QApplication, QLabel
from PyQt6.uic import loadUi

class Camera:
    def __init__(self):
        self.mainui = loadUi('window.ui')
        self.mainui.show()

        self.mainui.pb_fechar.clicked.connect(self.close_camera)

        # Seleção de câmeras disponíveis
        self.available_cameras = self.find_available_cameras(max_index=5) or [0]
        for index in self.available_cameras:
            self.mainui.comboBox.addItem(f"Câmera {index}", index)

        self.current_camera_index = self.available_cameras[0]
        self.mainui.comboBox.currentIndexChanged.connect(self.change_camera_index)
        self.mainui.comboBox.setCurrentIndex(0)

        # Captura da câmera
        self.cap = cv2.VideoCapture(self.current_camera_index)
        if not self.cap.isOpened():
            print("Não foi possível acessar a câmera.")
            return

        # Configuração do modelo de emoção
        self.detector_rostos = None
        self.session = None
        self.input_name = None
        self.emocoes = None
        self.clahe = None

        face_model_path = "ONNXs/face_detection_yunet_2023mar.onnx"
        emotion_model_path = "ONNXs/emotion-ferplus-12-int8.onnx"
        if os.path.exists(face_model_path) and os.path.exists(emotion_model_path):
            try:
                self.detector_rostos = cv2.FaceDetectorYN.create(face_model_path, "", (0, 0))
                self.session = onnxruntime.InferenceSession(emotion_model_path)
                self.input_name = self.session.get_inputs()[0].name
                self.emocoes = ['Neutro', 'Feliz', 'Surpreso', 'Triste', 'Bravo', 'Nojo', 'Medo', 'Deboche']
                self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                print("Modelos de detecção carregados com sucesso.")
            except Exception as e:
                print(f"Erro ao carregar modelos: {e}")
        else:
            print("Modelos ONNX não encontrados. Executando apenas com vídeo da câmera.")

        # Exibição de vídeo na interface
        self.video_label = QLabel()
        self.video_label.setScaledContents(True)
        self.video_label.setFixedSize(640, 480)
        self.mainui.verticalLayout_2.addWidget(self.video_label)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    def find_available_cameras(self, max_index=5):
        available = []
        for index in range(max_index + 1):
            cap = cv2.VideoCapture(index)
            if cap.isOpened():
                available.append(index)
                cap.release()
        return available

    def change_camera_index(self, combo_index):
        if combo_index < 0:
            return
        camera_index = self.mainui.comboBox.itemData(combo_index)
        if camera_index is None:
            return
        if camera_index == self.current_camera_index:
            return
        self.current_camera_index = camera_index

        if self.cap is not None and self.cap.isOpened():
            self.cap.release()

        self.cap = cv2.VideoCapture(self.current_camera_index)
        if not self.cap.isOpened():
            print(f"Não foi possível acessar a câmera {self.current_camera_index}.")

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        # Processamento de vídeo e detecção de emoção
        if self.detector_rostos is not None and self.session is not None:
            altura, largura = frame.shape[:2]
            self.detector_rostos.setInputSize((largura, altura))
            _, faces = self.detector_rostos.detect(frame)

            if faces is not None:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray_clahe = self.clahe.apply(gray)
                for rosto in faces:
                    x, y, w, h = map(int, rosto[:4])
                    x, y = max(0, x), max(0, y)
                    x2, y2 = min(largura, x + w), min(altura, y + h)
                    cv2.rectangle(frame, (x, y), (x2, y2), (255, 0, 0), 2)

                    face_crop = gray_clahe[y:y2, x:x2]
                    face_resized = cv2.resize(face_crop, (64, 64))
                    face_processed = np.array(face_resized, dtype=np.float32).reshape(1, 1, 64, 64)
                    predicoes = self.session.run(None, {self.input_name: face_processed})
                    emocao_detectada = self.emocoes[np.argmax(predicoes[0])]
                    cv2.putText(frame, emocao_detectada, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        # Conversão para exibição no PyQt
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        self.video_label.setPixmap(pixmap)
        self.video_label.repaint()

    def close_camera(self):
        self.timer.stop()
        self.cap.release()
        self.mainui.close()

class Home:
    def __init__(self):
        self.mainui = loadUi('splash.ui')
        self.camera = None
        
        # Conectar o botão pb_webcam ao método que abre a camera
        self.mainui.pb_webcam.clicked.connect(self.abrir_camera)
        self.mainui.pb_webcam.clicked.connect(self.mainui.close)
        
        self.mainui.show()
    
    def abrir_camera(self):
        """Abre a janela da câmera com detecção de emoção"""
        self.camera = Camera()


if __name__ == "__main__":
    
    app = QApplication([])
    main = Home()
    app.exec()