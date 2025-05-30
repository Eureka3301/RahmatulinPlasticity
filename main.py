import cv2
import numpy as np
import matplotlib.pyplot as plt

# Функция скользящего среднего
def moving_average(data, window_size=5):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

# === Загрузка и предобработка ===
image_path = "1.jpg"
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError("Не удалось загрузить изображение.")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Усиление контраста
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
contrast = clahe.apply(gray)

# Порог Otsu
_, thresh = cv2.threshold(contrast, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Поиск самого большого контура
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if not contours:
    raise ValueError("Контуры не найдены.")

contour = max(contours, key=cv2.contourArea)

# Аппроксимация контура для сглаживания
epsilon = 0.0001 * cv2.arcLength(contour, True)
smoothed_contour = cv2.approxPolyDP(contour, epsilon, True)

image_with_contour = image.copy()
cv2.drawContours(image_with_contour, [smoothed_contour], -1, (0, 255, 0), 2)

# Измерение ширины по высоте
x, y, w, h = cv2.boundingRect(smoothed_contour)
top_width_px = None
heights = []
diameters_mm = []

for i in range(y, y + h):
    row_points = smoothed_contour[smoothed_contour[:, 0, 1] == i]
    if len(row_points) >= 2:
        xs = row_points[:, 0, 0]
        left, right = xs.min(), xs.max()
        width_px = right - left
        if top_width_px is None:
            top_width_px = width_px  # Используется для калибровки
        width_mm = (width_px / top_width_px) * 8.0  # 8 мм — реальный верхний диаметр
        heights.append(i - y)
        diameters_mm.append(width_mm)

def moving_max_with_positions(data, positions, window_size=7):
    max_values = []
    max_positions = []
    length = len(data)
    half_win = window_size // 2

    for i in range(half_win, length - half_win):
        window = data[i - half_win : i + half_win + 1]
        window_pos = positions[i - half_win : i + half_win + 1]
        max_val = max(window)
        
        # Позиция для max — выбираем позицию центрального элемента окна (i)
        max_values.append(max_val)
        max_positions.append(positions[i])

    return np.array(max_values), np.array(max_positions)

# Допустим, у тебя есть diameters_mm и heights

window_size = 3  # обязательно нечетное для симметрии окна
diameters_max, heights_max = moving_max_with_positions(diameters_mm, heights, window_size)

# Теперь длины diameters_max и heights_max совпадают, и можно рисовать:
plt.figure(figsize=(6, 4))
plt.plot(heights, diameters_mm, color='blue', alpha=0.5, label='Исходные данные')
plt.plot(heights_max, diameters_max, color='green', label='Максимум по окну')
plt.gca().invert_xaxis()
plt.title("Диаметр по высоте с аппроксимацией и скользящим максимумом")
plt.xlabel("Высота (пиксели от верха)")
plt.ylabel("Диаметр (мм)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()