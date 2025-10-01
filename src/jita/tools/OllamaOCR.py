import ollama
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