import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Крок 1: Збір даних
data_dir = "dataset"
classes = ["apples", "plums"]
num_classes = len(classes)
image_width, image_height = 128, 128
num_channels = 3

images = []
labels = []

# Завантаження зображень та міток
for idx, class_name in enumerate(classes):
    class_dir = os.path.join(data_dir, class_name)
    for img_name in os.listdir(class_dir):
        img_path = os.path.join(data_dir, class_name, img_name)  # Змінений рядок
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (image_width, image_height))
        images.append(img)
        labels.append(idx)

# Перетворення зображень та міток в numpy-масиви
images = np.array(images)
labels = np.array(labels)

# Крок 2: Підготовка даних
# Розподіл даних на тренувальний та тестовий набори
train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2)

# Нормалізація пікселів зображень
train_images = train_images / 255.0
test_images = test_images / 255.0

# Перетворення міток у формат one-hot encoding
train_labels = tf.keras.utils.to_categorical(train_labels, num_classes)
test_labels = tf.keras.utils.to_categorical(test_labels, num_classes)

# Крок 4: Побудова моделі
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(image_width, image_height, num_channels)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Крок 5: Тренування моделі
model.fit(train_images, train_labels, epochs=10, batch_size=32)

# Крок 6: Оцінка моделі
loss, accuracy = model.evaluate(test_images, test_labels)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)

# Графічний інтерфейс
root = tk.Tk()

# Функція для вибору нового зображення
def select_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
    if file_path:
        img = Image.open(file_path)
        img = img.resize((image_width, image_height), Image.ANTIALIAS)
        img = ImageTk.PhotoImage(img)
        image_label.configure(image=img)
        image_label.image = img

        # Класифікація нового зображення
        new_image = cv2.imread(file_path)
        new_image = cv2.resize(new_image, (image_width, image_height))
        new_image = np.expand_dims(new_image, axis=0)
        new_image = new_image / 255.0
        predictions = model.predict(new_image)
        class_index = np.argmax(predictions)
        predicted_class = classes[class_index]
        prediction_label.configure(text="Predicted Class: " + predicted_class)

# Кнопка для вибору зображення
select_button = tk.Button(root, text="Select Image", command=select_image)
select_button.pack()

# Відображення зображення
image_label = tk.Label(root)
image_label.pack()

# Відображення передбаченого класу
prediction_label = tk.Label(root)
prediction_label.pack()

root.mainloop()