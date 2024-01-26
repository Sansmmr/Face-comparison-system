import dlib
import numpy as np
from skimage import io
from scipy.spatial import distance
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk, ImageDraw

import pyodbc

# Параметри підключення до SQL Server
# server = 'DESKTOP-4MGRCV0'
# database = 'SCAMERS'

# # Строка підключення
# conn_str = f'DRIVER={{SQL Server}};SERVER={server};DATABASE={database};'

# # Підключення до SQL Server
# conn = pyodbc.connect(conn_str)

# # Створення курсора для виконання SQL-запитів
# cursor = conn.cursor()



# # Закриття підключення
# conn.close()



class FaceDetector:
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()

    def detect_faces(self, image_np):
        return self.detector(image_np, 1)


class FaceLandmarkDetector:
    def __init__(self, shape_predictor_path):
        self.shape_predictor = dlib.shape_predictor(shape_predictor_path)

    def detect_landmarks(self, image_np, face):
        return self.shape_predictor(image_np, face)


class FaceRecognizer:
    def __init__(self, face_recognizer_path):
        self.face_recognizer = dlib.face_recognition_model_v1(face_recognizer_path)

    def compute_face_descriptor(self, image_np, shape):
        return self.face_recognizer.compute_face_descriptor(image_np, shape)


class FaceComparatorGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Метод автоматизованого визначення ідентичності людини за фотографічним зображенням з використанням глибокої згорткової нейронної мережі")

        self.frame = tk.Frame(self.root)
        self.frame.pack(pady=10)

        self.img_label1 = tk.Label(self.frame)
        self.img_label1.pack(side=tk.LEFT, padx=10)

        self.img_label2 = tk.Label(self.frame)
        self.img_label2.pack(side=tk.LEFT, padx=10)

        self.btn_load1 = tk.Button(self.root, text="Завантажити перше зображення", command=self.load_image1)
        self.btn_load1.pack(side=tk.LEFT, padx=10)

        self.btn_load2 = tk.Button(self.root, text="Завантажити друге зображення", command=self.load_image2)
        self.btn_load2.pack(side=tk.LEFT, padx=10)

        self.result_label = tk.Label(self.root, text="Результат:")
        self.result_label.pack()

        self.identity_label = tk.Label(self.root, text="")
        self.identity_label.pack()

        # self.addres_crime_entry=tk.Entry(self.root,text="Введіть id злочину")
        # self.addres_crime_entry.pack()

        # submit_button = tk.Button(self.root, text="Отримати", command=self.get_input) 
        # submit_button.pack()

        self.face_detector = FaceDetector()
        self.face_landmark_detector = FaceLandmarkDetector('D:/Directory/shape_predictor_68_face_landmarks.dat')
        self.face_recognizer = FaceRecognizer('D:/Directory/dlib_face_recognition_resnet_model_v1.dat')

        self.img1 = None
        self.img2 = None
        self.shape1 = None
        self.shape2 = None
        self.distance_value = 0.0  # Додати змінну для збереження значення коефіцієнта похожості

        self.root.mainloop()


    def load_image1(self):
        


        file_path = filedialog.askopenfilename()
        # file_path=results[0][0]
        if file_path:
            self.img1 = Image.open(file_path)
            self.img1 = self.img1.resize((300, 300), Image.LANCZOS)
            img1_tk = ImageTk.PhotoImage(self.img1)
            self.img_label1.config(image=img1_tk)
            self.img_label1.image = img1_tk
            img1_np = np.array(self.img1.convert('RGB'))
            dets1 = self.face_detector.detect_faces(img1_np)
            if len(dets1) > 0:
                d = dets1[0]
                self.shape1 = self.face_landmark_detector.detect_landmarks(img1_np, d)
                img_with_landmarks = self.draw_landmarks(self.img1.copy(), self.shape1)
                img_tk = ImageTk.PhotoImage(img_with_landmarks)
                self.img_label1.config(image=img_tk)
                self.img_label1.image = img_tk
                self.compute_distance()
            else:
                self.shape1 = None
                print("На вибраному зображенні не знайдено обличчя")

    def load_image2(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.img2 = Image.open(file_path)
            self.img2 = self.img2.resize((300, 300), Image.LANCZOS)
            img2_tk = ImageTk.PhotoImage(self.img2)
            self.img_label2.config(image=img2_tk)
            self.img_label2.image = img2_tk
            img2_np = np.array(self.img2.convert('RGB'))
            dets2 = self.face_detector.detect_faces(img2_np)
            if len(dets2) > 0:
                d = dets2[0]
                self.shape2 = self.face_landmark_detector.detect_landmarks(img2_np, d)
                img_with_landmarks = self.draw_landmarks(self.img2.copy(), self.shape2)
                img_tk = ImageTk.PhotoImage(img_with_landmarks)
                self.img_label2.config(image=img_tk)
                self.img_label2.image = img_tk
                self.compute_distance()
            else:
                self.shape2 = None
                print("На вибраному зображенні не знайдено обличчя")

    def draw_landmarks(self, image, shape):
        draw = ImageDraw.Draw(image)
        points = shape.parts()
        num_points = shape.num_parts
        for i in range(num_points - 1):
            x1, y1 = points[i].x, points[i].y
            x2, y2 = points[i + 1].x, points[i + 1].y
            draw.line((x1, y1, x2, y2), fill='red', width=2)

        x1, y1 = points[0].x, points[0].y
        x2, y2 = points[num_points - 1].x, points[num_points - 1].y
        draw.line((x1, y1, x2, y2), fill='red', width=2)
        return image

    def compute_distance(self):
        if self.img1 is not None and self.img2 is not None and self.shape1 is not None and self.shape2 is not None:
            try:
                face_descriptor1 = self.face_recognizer.compute_face_descriptor(np.array(self.img1), self.shape1)
                face_descriptor2 = self.face_recognizer.compute_face_descriptor(np.array(self.img2), self.shape2)
                self.distance_value = distance.euclidean(face_descriptor1, face_descriptor2)
                self.result_label.config(text="Результат: {:.2f}".format(self.distance_value))
                if self.distance_value < 0.5:
                    self.identity_label.config(text="Люди ідентичні")
                else:
                    self.identity_label.config(text="Люди не ідентичні")
            except:
                self.result_label.config(text="На одному або обох зображеннях не знайдено обличчя")
                self.identity_label.config(text="")
        else:
            self.result_label.config(text="Деякі зображення не були завантажені")
            self.identity_label.config(text="")

    def create_additional_tab(self):
        # Метод для створення додаткової вкладки
        additional_tab = tk.Toplevel(self.root)
        additional_tab.title("Додаткова вкладка")

        label = tk.Label(additional_tab, text="Введіть дані для пошуку обличчя:")
        label.pack()

        search_entry = tk.Entry(additional_tab)
        search_entry.pack()

        search_button = tk.Button(additional_tab, text="Пошук", command=self.search_face)
        search_button.pack()
                    
#     def get_input(self):
#         addres_crime=self.addres_crime_entry.get()
#         print(addres_crime)
#         # Приклад SQL-запиту
                       
#         cursor.execute("SELECT * FROM crime WHERE addres = ?" , addres_crime)

# # # Отримання результатів запиту
#         results = cursor.fetchall()
#         print(results[0][2])
#         cursor.execute("SELECT * FROM detectives WHERE addres = ?" , addres_crime)
        
FaceComparatorGUI()


