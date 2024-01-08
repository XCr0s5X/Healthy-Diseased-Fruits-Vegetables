import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# Крок 1: Збір даних
data_dir = "dataset/train"
classes = ["apples", "plums"]
# Змінна, яка містить класи здорових фруктів
healthy_classes = ["healthy", "healthy"]
num_classes = len(classes)
image_width, image_height = 128, 128
num_channels = 3

images = []
labels = []

# Завантаження зображень та міток
for idx, class_name in enumerate(classes):
    class_dir = os.path.join(data_dir, class_name)
    for subclass_name in os.listdir(class_dir):
        subclass_dir = os.path.join(class_dir, subclass_name)
        if os.path.isdir(subclass_dir):  # Перевірка, чи є це директорія
            # Визначаємо, чи належить поточний підкаталог до здорових фруктів
            is_healthy = subclass_dir in healthy_classes
            for img_name in os.listdir(subclass_dir):
                img_path = os.path.join(subclass_dir, img_name)
                if os.path.isfile(img_path):  # Перевірка, чи є це файл
                    img = cv2.imread(img_path)
                    img = cv2.resize(img, (image_width, image_height))
                    images.append(img)
                    labels.append((idx, is_healthy))

# Перетворення зображень та міток в numpy-масиви
images = np.array(images)
labels = np.array(labels)

# Розподіл міток на окремі змінні
class_indices, health_labels = zip(*labels)
class_indices = np.array(class_indices)
health_labels = np.array(health_labels)

# Код для розподілу даних на тренувальний та тестовий набори
train_images, test_images, train_class_indices, test_class_indices, train_health_labels, test_health_labels = train_test_split(
    images, class_indices, health_labels, test_size=0.2, stratify=class_indices, random_state=42)

# Перетворення міток у формат one-hot encoding
train_labels = tf.keras.utils.to_categorical(train_class_indices, num_classes)
test_labels = tf.keras.utils.to_categorical(test_class_indices, num_classes)

# Крок 4: Побудова моделі
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(image_width, image_height, num_channels))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# Заморожуємо базову модель
for layer in base_model.layers:
    layer.trainable = False

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
        prediction = model.predict(new_image)
        class_index = np.argmax(prediction)
        predicted_class = classes[class_index]

        # Визначення типу фрукта та стану
        fruit_type = classes[class_index]
        is_healthy = False

        if fruit_type.startswith("apples"):
            if predicted_class.endswith("healthy"):
                is_healthy = True
        elif fruit_type.startswith("plums"):
            if predicted_class.endswith("healthy"):
                is_healthy = True

        # ...

        # Відображення результатів
        if fruit_type is not None:
            prediction_label.configure(text="Predicted Class: " + fruit_type)
            status_label.configure(text="Status: " + ("Healthy" if is_healthy else "Diseased"))
        else:
            prediction_label.configure(text="Predicted Class: Unknown")
            status_label.configure(text="Status: Unknown")

# Кнопка для вибору зображення
select_button = tk.Button(root, text="Select Image", command=select_image)
select_button.pack()

# Відображення зображення
image_label = tk.Label(root)
image_label.pack()

# Відображення передбаченого класу
prediction_label = tk.Label(root)
prediction_label.pack()

# Відображення статусу
status_label = tk.Label(root)
status_label.pack()

root.mainloop()
