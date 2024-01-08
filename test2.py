import cv2
import tkinter as tk
from tkinter import filedialog
from tensorflow.keras.models import load_model
import os
import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from PIL import Image, ImageTk


# # Функція для навчання моделі
# def train_model():
#     # Шлях до директорії з навчальними даними
#     dataset_dir = 'dataset/train'

#     # Список класів (яблуко, слива)
#     classes = ['apples', 'plums']

#     # Загальна кількість зображень для навчання
#     total_images = 0

#     # Створення списків для зображень та міток
#     images = []
#     labels = []

#     # Перебір класів та підкласів
#     for class_name in classes:
#         class_dir = os.path.join(dataset_dir, class_name)

#         sub_classes = os.listdir(class_dir)
#         for sub_class in sub_classes:
#             sub_class_dir = os.path.join(class_dir, sub_class)
#             if os.path.isdir(sub_class_dir):
#                 image_files = os.listdir(sub_class_dir)
#                 for image_file in image_files:
#                     image_path = os.path.join(sub_class_dir, image_file)
#                     image = cv2.imread(image_path)
#                     image = cv2.resize(image, (224, 224))  # Попереднє масштабування зображення до потрібного розміру

#                     images.append(image)
#                     labels.append(classes.index(class_name))

#                     total_images += 1

#     # Конвертування списків в масиви NumPy
#     images = np.array(images, dtype='float32')
#     labels = np.array(labels)

#     # Нормалізація зображень до діапазону [0, 1]
#     images /= 255.0

#     # Розділення даних на тренувальний та тестувальний набори
#     from sklearn.model_selection import train_test_split
#     X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

#     # Будування та навчання моделі (використовуйте відповідну архітектуру моделі для вашої задачі)

#     base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

#     model = Sequential()
#     model.add(base_model)
#     model.add(GlobalAveragePooling2D())
#     model.add(Dense(256, activation='relu'))
#     model.add(Dropout(0.5))
#     model.add(Dense(len(classes), activation='softmax'))

#     model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#     model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

#     # Збереження навченої моделі на диск
#     model.save('fruit_model.h5')

# train_model()

# Завантаження навченої моделі
model = load_model('fruit_model.h5')

# Функція для розпізнавання фруктів на завантаженому зображенні
def recognize_fruit(image_path):
    # Завантаження зображення
    image = cv2.imread(image_path)
    
    # Підготовка зображення перед передачею в модель
    image = cv2.resize(image, (224, 224))  # Попереднє масштабування зображення до потрібного розміру
    image = image.astype('float32') / 255.0

    # Додавання додаткового розміру пакунку (batch dimension)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))

    # Передача зображення в модель для розпізнавання
    predictions = model.predict(image)

    # Визначення виду фрукта
    classes = ['яблуко', 'слива']
    class_index = predictions.argmax()
    fruit_type = classes[class_index]

    # Визначення, чи є фрукт здоровим (поки що рандомно генерується)
    is_healthy = bool(predictions[0][class_index] > 0.9)

    # Показ результатів розпізнавання
    result_text = f"Вид фрукта: {fruit_type}\nЗдоровий: {'Так' if is_healthy else 'Ні'}"

    # Оновлення мітки з результатами
    result_label.config(text=result_text)



# Функція для обробки натискання кнопки "Завантажити зображення"
def browse_image():
    # Відкриття діалогового вікна вибору файлу
    file_path = filedialog.askopenfilename(filetypes=[("Зображення", "*.jpg;*.jpeg;*.png")])

    # Передача шляху до зображення функції розпізнавання фруктів
    recognize_fruit(file_path)

    # Відображення зображення
    image = Image.open(file_path)
    image.thumbnail((200, 200))  # Зміна розміру зображення для відображення
    photo = ImageTk.PhotoImage(image)
    image_label.config(image=photo)
    image_label.image = photo  # Збереження посилання на зображення, щоб воно не було видалено зі сміття Python

# Створення головного вікна програми
window = tk.Tk()
window.title("Розпізнавання фруктів")
window.geometry("400x400")

# Мітка для відображення зображення
image_label = tk.Label(window)
image_label.pack()

# Кнопка для завантаження зображення
browse_button = tk.Button(window, text="Завантажити зображення", command=browse_image)
browse_button.pack(pady=20)

# Мітка для відображення результатів розпізнавання
result_label = tk.Label(window, text="")
result_label.pack()

# Запуск головного циклу GUI
window.mainloop()