import cv2
import numpy as np
import matplotlib.pyplot as plt

def process_image(image_path, reference_height_mm=8):
    # Загрузка изображения
    img = cv2.imread(image_path)
    if img is None:
        print("Ошибка загрузки изображения")
        return
    
    # Преобразование в градации серого
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Бинаризация (можно настроить параметры)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Поиск контуров
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print("Контуры не найдены")
        return
    
    # Выбор самого большого контура
    main_contour = max(contours, key=cv2.contourArea)
    
    # Находим ограничивающий прямоугольник
    x, y, w, h = cv2.boundingRect(main_contour)
    
    # Создаем маску для контура
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [main_contour], 0, 255, -1)
    
    # Определяем верхнюю часть для калибровки (первые 10% высоты)
    calibration_height = int(h * 0.1)
    top_part = mask[y:y+calibration_height, x:x+w]
    
    # Находим ширину в верхней части (в пикселях)
    top_width_px = 0
    for col in range(top_part.shape[1]):
        if np.any(top_part[:, col] > 0):
            left = col
            break
    for col in range(top_part.shape[1]-1, -1, -1):
        if np.any(top_part[:, col] > 0):
            right = col
            break
    top_width_px = right - left
    
    # Коэффициент калибровки (мм на пиксель)
    px_to_mm = reference_height_mm / top_width_px
    
    # Анализ изменения ширины по высоте
    width_profile = []
    heights = []
    
    for row in range(y, y+h):
        # Находим левую и правую границы для текущей строки
        line = mask[row, x:x+w]
        if not np.any(line > 0):
            continue
        
        left = np.argmax(line > 0)
        right = len(line) - np.argmax(line[::-1] > 0) - 1
        width_px = right - left
        width_mm = width_px * px_to_mm
        
        width_profile.append(width_mm)
        heights.append((row - y) * px_to_mm)  # Высота в мм
    
    def moving_average(data, window_size=5):
        return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

    window_size = 11
    width_profile = moving_average(width_profile, window_size=window_size)
    heights = heights[(window_size//2): -(window_size//2)]

    # Возвращаем результаты
    return {
        'width_profile': width_profile,
        'heights': heights,
        'px_to_mm_ratio': px_to_mm,
        'contour': main_contour,
        'image_size': img.shape
    }

# Пример использования
result = process_image('1.jpg')

w0 = 8.0

w = np.array(result['width_profile'])
h = np.array(result['heights'])

exx = (w-w0)/w0

ux = np.array([np.trapz(exx[:i], h[:i]) for i in range(len(exx))])

H = h + ux

# Производная eini по h
dexx_dH = np.gradient(exx, H)

L = 20.0

K = dexx_dH/(1-(H/(2*L-H))**2)

sc = 310e+6
E = 128e+9
ec = sc/E

em = ec - np.array([np.trapz(K[i:], H[i:]) for i in range(len(exx))])

sm = E*(em-exx)

import matplotlib.pyplot as plt

# --- Построение всех графиков в сетке ---
fig, axs = plt.subplots(2, 2, figsize=(12, 8))
axs = axs.flatten()

# 1. Профиль ширины
axs[0].plot(h, w, label="Ширина (мм)", color='blue')
axs[0].set_title("Профиль ширины w(h)")
axs[0].set_xlabel("Высота (мм)")
axs[0].set_ylabel("Ширина (мм)")
axs[0].grid(True)
axs[0].legend()

# 2. Эффективная деформация
axs[1].plot(H, sm, label="efin = 1 - w/8", color='green')
axs[1].set_title("Остаточная деформация")
axs[1].set_xlabel("Высота (мм)")
axs[1].set_ylabel("Деформация")
axs[1].grid(True)
axs[1].legend()

# 3. Производная деформации по высоте
axs[2].plot(h, dexx_dH, label="dexx_dH", color='red')
axs[2].set_title("Производная деформации по высоте")
axs[2].set_xlabel("Высота (мм)")
axs[2].set_ylabel("Производная")
axs[2].grid(True)
axs[2].legend()

# 4. Оставим пустым или добавим при необходимости
axs[3].plot(em, sm, label="s-e", color='red')
axs[3].set_title("Диаграмма (динамическая)")
axs[3].set_xlabel("Деформация")
axs[3].set_ylabel("Напряжение")
axs[3].grid(True)
axs[3].legend()

plt.tight_layout()
plt.show()
