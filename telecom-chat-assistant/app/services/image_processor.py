import os
import time
import base64
from pathlib import Path
from PIL import Image
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

from app.core.config import settings
from app.services.document_processor import DocumentProcessor


class ImageProcessor:
    """Process images using vision-capable models (Singleton)"""

    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        if settings.USE_LOCAL_MODEL:
            print(f"Initializing Ollama vision model: {settings.LOCAL_VISION_MODEL_NAME}")
            self.vision_model = ChatOllama(
                model=settings.LOCAL_VISION_MODEL_NAME,
                base_url=settings.OLLAMA_BASE_URL,
                temperature=0.3,
                num_ctx=2048,
            )
        else:
            print(f"Initializing OpenAI vision model: {settings.VISION_MODEL}")
            self.vision_model = ChatOpenAI(
                model_name=settings.VISION_MODEL,
                temperature=0.3,
                max_tokens=1000
            )

        self._initialized = True
        print("ImageProcessor initialized (Singleton)")

    def resize_image_if_needed(self, image_path: str) -> str:
        """Resize image if it's too large"""
        try:
            img = Image.open(image_path)

            max_dim = settings.IMAGE_RESIZE_MAX_DIMENSION
            if max(img.size) > max_dim:
                print(f"Resizing image from {img.size} to fit {max_dim}px")

                ratio = max_dim / max(img.size)
                new_size = tuple([int(x * ratio) for x in img.size])

                img = img.resize(new_size, Image.Resampling.LANCZOS)

                image_name = os.path.splitext(os.path.basename(image_path))[0]
                resized_path = os.path.join(settings.TMP_DATA_DIR, f"{image_name}_resized.jpg")
                img.save(resized_path, "JPEG", quality=85, optimize=True)

                return resized_path

            return image_path
        except Exception as e:
            print(f"Resize failed: {str(e)}, using original")
            return image_path

    def analyze_image(self, image_path: str, query: str = None) -> str:
        """Analyze image and extract information"""

        processed_image_path = self.resize_image_if_needed(image_path)

        try:
            if settings.USE_LOCAL_MODEL:
                result = self._analyze_with_ollama(processed_image_path, query)
            else:
                result = self._analyze_with_openai(processed_image_path, query)

            if processed_image_path != image_path and os.path.exists(processed_image_path):
                os.remove(processed_image_path)

            return result

        except Exception as e:
            error_msg = str(e)
            print(f"Vision model error: {error_msg}")

            if processed_image_path != image_path and os.path.exists(processed_image_path):
                os.remove(processed_image_path)

            if settings.USE_OCR_FALLBACK:
                print("Falling back to OCR...")
                return self._fallback_ocr(image_path, query)
            else:
                return f"Error analyzing image: {error_msg}"

    def _fallback_ocr(self, image_path: str, query: str = None) -> str:
        """Fallback OCR analysis"""
        try:
            ocr_text = DocumentProcessor.process_image_with_ocr(image_path)

            if not ocr_text.strip():
                return "No text could be extracted from the image using OCR."

            result = f"Text extracted from image using OCR:\n\n{ocr_text}"

            if query:
                result += f"\n\nRegarding your question: '{query}'\n"
                result += "The above text was extracted from the image. Please let me know if you need specific information from this text."

            return result
        except Exception as e:
            return f"Both vision model and OCR failed. Error: {str(e)}"

    def _analyze_with_openai(self, image_path: str, query: str = None) -> str:
        """Analyze image using OpenAI GPT-4 Vision"""
        base64_image = DocumentProcessor.encode_image_to_base64(image_path)

        if not base64_image:
            raise ValueError("Failed to encode image")

        ext = Path(image_path).suffix.lower()
        mime_type = {
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.webp': 'image/webp',
            '.gif': 'image/gif'
        }.get(ext, 'image/jpeg')

        default_query = """قم بتحليل هذه الصورة بالتفصيل واستخرج جميع المعلومات منها:

Please analyze this image in detail and extract all information:

1. أي نصوص ظاهرة (عربي وإنجليزي) / Any visible text (Arabic and English)
2. وصف العناصر المرئية / Description of visual elements
3. أي جداول أو رسوم بيانية أو بيانات منظمة / Any tables, charts, or structured data
4. السياق والغرض من الصورة / Context and purpose of the image
5. معلومات عن الباقات أو الخدمات إن وجدت / Information about packages or services if present

قدم المعلومات بتنسيق واضح ومنظم.
Provide the information in a clear, organized format."""

        user_query = query if query else default_query

        message = HumanMessage(
            content=[
                {"type": "text", "text": user_query},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{base64_image}",
                        "detail": "high"
                    }
                }
            ]
        )

        response = self.vision_model.invoke([message])
        return response.content

    def _analyze_with_ollama(self, image_path: str, query: str = None) -> str:
        """Analyze image using Ollama vision model"""

        with open(image_path, "rb") as img_file:
            image_data = img_file.read()

        base64_image = base64.b64encode(image_data).decode('utf-8')

        default_query = """قم بتحليل هذه الصورة واستخرج جميع المعلومات منها.

Analyze this image and extract all information from it.

تضمين:
Include:
1. أي نص مرئي (عربي وإنجليزي) / Any visible text (Arabic and English)
2. وصف للعناصر المرئية / Description of visual elements
3. معلومات عن الباقات أو الخدمات / Information about packages or services
4. أي أسعار أو أرقام / Any prices or numbers

قدم المعلومات بوضوح.
Provide information clearly."""

        user_query = query if query else default_query

        try:
            message = HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": user_query
                    },
                    {
                        "type": "image_url",
                        "image_url": f"data:image/jpeg;base64,{base64_image}"
                    }
                ]
            )

            max_retries = 2
            for attempt in range(max_retries):
                try:
                    response = self.vision_model.invoke([message])
                    return response.content
                except Exception as e:
                    if attempt < max_retries - 1:
                        print(f"Attempt {attempt + 1} failed, retrying...")
                        time.sleep(2)
                    else:
                        raise e

        except Exception as e:
            error_str = str(e)
            if "500" in error_str or "runner" in error_str.lower():
                raise Exception(
                    f"Ollama vision model crashed (likely out of memory). "
                )
            else:
                raise e