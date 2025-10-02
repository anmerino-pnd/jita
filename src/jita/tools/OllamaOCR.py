import os
import cv2
import math
import ollama
import numpy as np
from PIL import Image

from jita.abstract_tools.supermarket_ocr import SupermarketOCRBase

class OllamaOCR(SupermarketOCRBase):
    def __init__(self, model: str):
        self.model = model

    def extract_text(self, image_path):
        return ollama.chat(
    model=self.model,
    messages=[
          {
              'role': 'system',
              'content': self.system_prompt,
              'role': 'user',
              'content': "Analiza este folleto",
              'images': [image_path]  # 👈 lista de rutas de imagen
          }
      ],
    options={'temperature': 0, 'format': 'json'}
  )['message']['content']
    
    def smart_slicing(self, image_path, folder_path):
        """
        Divide una imagen en cuadrados calculando automáticamente el tamaño más eficiente.
        """
        img = Image.open(image_path)
        width, height = img.size
        base_name = os.path.splitext(os.path.basename(image_path))[0]

        # 1. calcular el divisor común más grande de width y height
        gcd = math.gcd(width, height)

        # 2. elegir tile_size:
        #    - si gcd es suficientemente grande, usarlo (división perfecta)
        #    - si no, aproximar usando la dimensión menor
        if gcd >= min(width, height) // 3:
            tile_size = gcd
        else:
            tile_size = min(width, height) // 2  # heurística: 2 cortes como mínimo

        # 3. calcular número de tiles
        n_cols = math.ceil(width / tile_size)
        n_rows = math.ceil(height / tile_size)

        tiles = []
        for row in range(n_rows):
            for col in range(n_cols):
                left = col * tile_size
                top = row * tile_size
                right = min(left + tile_size, width)
                bottom = min(top + tile_size, height)

                tile = img.crop((left, top, right, bottom))
                filename = f"{base_name}_r{row}_c{col}.png"
                path_out = os.path.join(folder_path, filename)
                tile.save(path_out, quality=100)
                tiles.append(path_out)

        return tiles

    def smart_crop(self, image_path, output_folder, min_area=5000):
        os.makedirs(output_folder, exist_ok=True)
        
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 1. Binarización adaptativa
        thresh = cv2.adaptiveThreshold(gray, 255,
                                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, 25, 15)

        # 2. Encontrar contornos
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        crops = []
        for i, cnt in enumerate(contours):
            x, y, w, h = cv2.boundingRect(cnt)

            # 3. Filtrar contornos pequeños
            if w * h > min_area:
                crop = img[y:y+h, x:x+w]
                out_path = os.path.join(output_folder, f"crop_{i}.png")
                cv2.imwrite(out_path, crop)
                crops.append((x, y, w, h, out_path))
        
        return crops
        
    def smart_crop_auto_area(self, image_path, output_folder, area_ratio=0.01):
        """
        Recorta automáticamente bloques de una imagen de folleto usando OpenCV.
        `area_ratio` define el tamaño mínimo relativo del contorno respecto al área total.
        """        
        img = cv2.imread(image_path)
        height, width = img.shape[:2]
        total_area = width * height

        # Calculamos min_area dinámicamente
        min_area = total_area * area_ratio

        # Convertir a gris y binarizar
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 25, 15
        )

        # Encontrar contornos externos
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        crops = []
        for i, cnt in enumerate(contours):
            x, y, w, h = cv2.boundingRect(cnt)

            if w * h >= min_area:
                crop = img[y:y+h, x:x+w]
                out_path = os.path.join(output_folder, f"crop_{i}.png")
                cv2.imwrite(out_path, crop)
                crops.append((x, y, w, h, out_path))
        
        return crops