import cv2
import numpy as np
import matplotlib.pyplot as plt

def visualise(img, contour, thickness, heights):
    # Визуализация
    plt.figure(figsize=(10, 5))
    
    # 1. Исходное изображение с контуром
    plt.subplot(1, 2, 1)
    img_with_contour = img.copy()
    cv2.drawContours(img_with_contour, [contour], -1, (0, 255, 0), 2)
    plt.imshow(cv2.cvtColor(img_with_contour, cv2.COLOR_BGR2RGB))
    plt.title('Контур образца')
    plt.axis('off')
    
    # 2. График толщины
    plt.subplot(1, 2, 2)
    plt.plot(thickness, heights)
    plt.gca().invert_yaxis()  # Инвертируем ось Y
    plt.xlabel('Толщина (пиксели)')
    plt.ylabel('Высота (пиксели)')
    plt.title('Толщина образца по длине')
    plt.grid()
    
    plt.tight_layout()
    plt.show()
    
    # Вывод результатов
    if thickness:
        print("\nРезультаты измерения:")
        print(f"Максимальная толщина: {max(thickness)} пикселей")
        print(f"Минимальная толщина: {min(thickness)} пикселей")
        print(f"Средняя толщина: {np.mean(thickness):.1f} пикселей")
    else:
        print("Ошибка: не удалось измерить толщину")

def measure_thickness(image_path):
    # Загрузка изображения
    img = cv2.imread(image_path)
    if img is None:
        print("Ошибка: не удалось загрузить изображение")
        return
    
    # Конвертация в grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Бинаризация (для темного объекта на светлом фоне)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Поиск контуров
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("Ошибка: контуры не найдены")
        return
    
    # Выбор самого большого контура
    contour = max(contours, key=cv2.contourArea)
    
    # Создание маски контура
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [contour], -1, 255, -1)
    
    # Получение границ контура
    x, y, w, h = cv2.boundingRect(contour)
    
    # Измерение толщины для каждой строки
    thickness = []
    heights = []
    
    for row in range(y, y + h):
        # Находим все точки контура в текущей строке
        line = mask[row, :]
        points = np.where(line == 255)[0]
        
        if len(points) >= 2:
            left = points[0]
            right = points[-1]
            thickness.append(right - left)
            heights.append(row)
    
    visualise(img, contour, thickness, heights)



if __name__ == "__main__":
    while True:
        image_path = input()
        try:
            if image_path == 'exit':
                break
            measure_thickness(image_path)
        except Exception as e:
            print(e)
        