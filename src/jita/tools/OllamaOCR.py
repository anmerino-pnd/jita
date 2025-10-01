import os
import ollama
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
    

    def slicing_window(self, image_path, folder_path, window_height=950, overlap=100):
        img = Image.open(image_path)
        width, height = img.size

        base_name = os.path.splitext(os.path.basename(image_path))[0]

        tiles = []
        step = window_height - overlap
        for top in range(0, height, step):
            bottom = min(top + window_height, height)
            tile = img.crop((0, top, width, bottom))

            filename = f"{base_name}_{top}_{bottom}.png"
            tile.save(f"{folder_path}/{filename}", quality=100)
            tiles.append(f"{folder_path}/{filename}")

            if bottom == height:
                break

        return tiles
