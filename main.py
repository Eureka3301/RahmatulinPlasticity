import cv2
import matplotlib.pyplot as plt
import numpy as np


def load_and_preprocess_image(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Не удалось загрузить изображение: {path}")
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if np.mean(binary) > 127:
        binary = cv2.bitwise_not(binary)

    return img, binary


def extract_object_mask(binary_image):
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("Контуры не найдены.")
    largest_contour = max(contours, key=cv2.contourArea)

    mask = np.zeros_like(binary_image)
    cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
    return mask


def measure_widths(mask, step=5):
    heights = []
    positions = []

    for y in range(0, mask.shape[0], step):
        row = mask[y, :]
        x_coords = np.where(row > 0)[0]
        if len(x_coords) > 0:
            width = int(x_coords[-1] - x_coords[0])
            heights.append(width)
            positions.append(y)

    return positions, heights


def get_bottom_width(mask, step=5):
    """Находит ширину объекта на самом нижнем горизонтальном срезе, где он ещё виден"""
    for y in range(mask.shape[0] - 1, 0, -step):
        row = mask[y, :]
        x_coords = np.where(row > 0)[0]
        if len(x_coords) > 0:
            return int(x_coords[-1] - x_coords[0])
    raise ValueError("Не удалось определить ширину нижнего края объекта.")


def draw_measurement_lines(image, mask, step=5):
    result = image.copy()
    for y in range(0, mask.shape[0], step):
        row = mask[y, :]
        x_coords = np.where(row > 0)[0]
        if len(x_coords) > 0:
            x_min, x_max = int(x_coords[0]), int(x_coords[-1])
            cv2.line(result, (x_min, y), (x_max, y), (0, 0, 255), 1)
    return result


def convert_to_millimeters(pixel_widths, reference_pixel_width, real_diameter_mm):
    pixels_per_mm = reference_pixel_width / real_diameter_mm
    return [w / pixels_per_mm for w in pixel_widths]


def plot_results(positions, widths_mm, image_with_lines):
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(positions, widths_mm, color='blue')
    plt.title("Изменение ширины вдоль длины")
    plt.xlabel("Положение по Y (пиксели)")
    plt.ylabel("Ширина (мм)")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(image_with_lines, cv2.COLOR_BGR2RGB))
    plt.title("Изображение с измерениями")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

def trim_measurement_region(positions, widths, drop_threshold=0.005):
    """Обрезает верх и низ на основе максимальной ширины и резкого падения внизу"""
    widths = np.array(widths)
    positions = np.array(positions)

    # 1. Найти индекс максимальной ширины
    max_index = np.argmax(widths)

    # 2. Найти резкое падение снизу — ищем, где ширина сильно падает по сравнению с предыдущей
    end_index = len(widths) - 1
    for i in range(len(widths) - 2, max_index, -1):
        if widths[i] < widths[i + 1] * (1 - drop_threshold):
            end_index = i + 1
            break

    # Вернуть усечённые массивы
    return positions[max_index:end_index + 1], widths[max_index:end_index + 1]

def main():
    path = 'photo_2025-05-22_00-11-32.jpg'
    real_diameter_mm = 5  # 🔧 Настоящий диаметр в нижней части

    img, binary = load_and_preprocess_image(path)
    mask = extract_object_mask(binary)
    positions, widths_px = measure_widths(mask, step=5)

    # 🔪 Автоматическая обрезка по Y
    trimmed_positions, trimmed_widths_px = trim_measurement_region(positions, widths_px)

    # 📏 Калибровка по ширине в НИЖНЕЙ точке после обрезки
    reference_pixel_width = trimmed_widths_px[-1]
    widths_mm = convert_to_millimeters(trimmed_widths_px, reference_pixel_width, real_diameter_mm)

    img_with_lines = draw_measurement_lines(img, mask, step=5)

    # Общая высота изображения в пикселях
    full_height = mask.shape[0]

    # Создаём массивы длиной в height (по числу пикселей по Y)
    width_mm_per_pixel = np.full(full_height, np.nan)
    length_mm_per_pixel = np.full(full_height, np.nan)

    # Шаг между строками, должен совпадать с step
    step = 5
    mm_per_step = real_diameter_mm / reference_pixel_width * step

    for i, y in enumerate(trimmed_positions):
        width_mm_per_pixel[y] = widths_mm[i]
        length_mm_per_pixel[y] = i * mm_per_step  # длина вдоль образца

    print("Y\tWidth_mm\tLength_mm")
    for y in range(full_height):
        if not np.isnan(width_mm_per_pixel[y]):
            print(f"{y}\t{width_mm_per_pixel[y]:.6f}\t{length_mm_per_pixel[y]:.6f}")


    plot_results(trimmed_positions, widths_mm, img_with_lines)



if __name__ == '__main__':
    main()
