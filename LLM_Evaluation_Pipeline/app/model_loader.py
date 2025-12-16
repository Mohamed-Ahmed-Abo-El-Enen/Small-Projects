import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelLoader:
    def __init__(self, model_name="Qwen/Qwen2-0.5B"):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        self.tokenizer = None
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the app and tokenizer"""
        try:
            logger.info(f"Loading app: {self.model_name}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True
            )
            
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            
            self.model.eval()
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading app: {e}")
            raise
    
    def generate(self, prompt, max_new_tokens=100, temperature=0.7):
        """Generate text from prompt"""
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_text = generated_text[len(prompt):].strip()
            
            return generated_text
            
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            raise
    
    def evaluate_mcq(self, question, choices, max_new_tokens=50, temperature=0.1):
        """Evaluate a multiple choice question"""
        prompt = f"""Answer the following multiple choice question by selecting the correct option (A, B, C, or D).
Only respond with the letter of your answer.

Question: {question}

Choices:
A. {choices.get('A', '')}
B. {choices.get('B', '')}
C. {choices.get('C', '')}
D. {choices.get('D', '')}

Answer:"""
        
        response = self.generate(prompt, max_new_tokens=max_new_tokens, temperature=temperature)
        
        answer = self._extract_answer(response)
        
        return {
            "answer": answer,
            "full_response": response
        }
    
    def _extract_answer(self, response):
        """Extract answer letter from app response"""
        response = response.strip().upper()
        
        for char in response:
            if char in ['A', 'B', 'C', 'D']:
                return char
        
        if response:
            return response[0] if response[0] in ['A', 'B', 'C', 'D'] else 'A'
        
        return 'A'
