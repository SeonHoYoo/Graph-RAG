import logging
import random
import torch
from typing import *
import torch.nn.functional as F

logger = logging.getLogger(__name__)
random.seed(26)


class FlanT5:
    def __init__(
        self, 
        model, 
        tokenizer
    ):
        self.model = model
        self.tokenizer = tokenizer
    
    
    def generate(
        self, 
        input: str, 
        **generator_args
    ) -> str:
        
        input_ids = self.tokenizer.encode(input, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(input_ids, **generator_args)
        answer = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

        return answer
    

    def generate_with_confidence(
        self,
        input: str,
        **generator_args
    ) -> Tuple[str, float]:

        input_ids = self.tokenizer.encode(input, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids,
                output_scores=True,
                return_dict_in_generate=True,
                **generator_args
            )

        answer = self.tokenizer.decode(generated_ids.sequences[0], skip_special_tokens=True).strip()

        scores = generated_ids.scores   # 각 토큰 step별 logits
        generated_tokens = generated_ids.sequences[0][1:]  # 첫 <pad> 제외

        confidences = []
        for i, token_id in enumerate(generated_tokens):
            if i >= len(scores):  # scores와 토큰 개수가 다를 수 있으므로 안전 체크
                break
            prob = F.softmax(scores[i], dim=-1)  # shape: [1, vocab_size]
            conf = prob[0, token_id].item()
            confidences.append(conf)

        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        return answer, avg_confidence 

    
class BaseModel(FlanT5):
    def __init__(
        self, 
        model, 
        tokenizer
    ):
        super().__init__(model, tokenizer)
        
        
    def get_infilling_prompt(
        self, 
        query: str, 
        evidence: Optional[str] = None
    ) -> str:
        
        if not evidence:
            prompt = f"Fill in the blank with the correct entity: {query}\nAnswer:"
        else:
            prompt = f"{evidence}\nBased on the above information, fill in the blank with the correct entity: {query}\nAnswer:"
        
        return prompt
    
    
    def infill(
        self, 
        query: str, 
        evidence: Optional[str] = None
    ) -> str:
        
        prompt = self.get_infilling_prompt(query, evidence)
        
        answer, conf = self.generate_with_confidence(prompt, max_new_tokens=32)
        if answer.lower().startswith("blank is "):
            answer = answer[len("blank is "):]
        print(prompt)
        print(answer)
        print('\n---\n')
        return answer, conf
    
    
    def parse_boolean_answer(
        self, 
        answer: str
    ) -> bool:
        
        answer = answer.split("\n")[0].lower().strip(" .")
        boolean_mapping = {
            "true": True, "false": False, "yes": True, "no": False,
            "it is impossible to say": False, "it's impossible to say": False,
            "it is impossible to tell": False, "it's impossible to tell": False,
            "it is not possible to say": False, "it's not possible to say": False,
            "it is not possible to tell": False, "it's not possible to tell": False
        }
        
        if answer in boolean_mapping:
            return boolean_mapping[answer]
        
        for sample_text, boolean_value in boolean_mapping.items():
            if answer.startswith(sample_text):
                return boolean_value
        
        logger.error(f"Unmapped answer detected: '{answer}'")
        return random.choice([True, False])
    
    
    def get_verification_prompt(
        self, 
        claim: str, 
        evidence: Optional[str] = None
    ) -> str:
        
        if not evidence:
            prompt = f"Claim: {claim}\nIs the claim true or false?\nAnswer:"
        else:
            prompt = f"Evidence: {evidence}\nClaim: {claim}\nIs the claim true or false?\nAnswer:"
        
        return prompt
    
    
    def verify(
        self, 
        claim: str, 
        evidence: Optional[str] = None
    ) -> bool:
        
        prompt = self.get_verification_prompt(claim, evidence)
        
        answer, conf = self.generate_with_confidence(prompt, max_new_tokens=8)
        answer = self.parse_boolean_answer(answer)
        
        print(prompt)
        print(answer)
        print('\n---\n')
        return answer, conf
    
    
    def get_answering_prompt(
        self, 
        question: str, 
        evidence: Optional[str] = None
    ) -> str:
        
        if not evidence:
            prompt = f"Question: {question}\nAnswer:"
        else:
            prompt = f"Context: {evidence}\nQuestion: {question}\nAnswer:"
        
        return prompt
    
    
    def generate_answer(
        self, 
        question: str, 
        evidence: Optional[str] = None, 
        **generator_args
    ) -> bool:
        
        prompt = self.get_answering_prompt(question, evidence)
        
        answer, conf = self.generate_with_confidence(prompt, **generator_args)
        print(prompt)
        print(answer)
        print('\n===\n')
        
        return answer, conf
    
    
    def generate_question(
        self, 
        triple_sent: str
    ) -> str:
        
        prompt = f"""Given the triplet, generate a question based on the triplet.  

Triplet: (Adams Township | is located in | a country: #1)
Question: In which country is Adams Township located?

Triplet: (an individual: #1 | composed | New York Counterpoint)
Question: Who composed New York Counterpoint?

Triplet: (Amalie Schoppe | died in | a date: #1)
Question: When did Amalie Schoppe die?

Triplet: (Sam Mangwana | is a citizen of | a country: #1)
Question: Of which country is Sam Mangwana a citizen?

Triplet: (the ball drop | started in | 1997 | in a state: #1)
Question: In which state did the ball drop start in 1997?

Triplet: ({triple_sent})
Question:"""
        
        question, conf = self.generate_with_confidence(prompt)
        
        print(f"Triplet: ({triple_sent})")
        print("Question:", question)
        print("\n---\n")
        
        return question, conf 