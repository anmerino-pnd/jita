import os
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
