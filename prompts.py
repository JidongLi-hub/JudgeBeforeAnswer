import os 
import json
from tqdm import tqdm

class Prompts:
    def __init__(self, q_type):
        self.supported_types = [
            "Entity Existence",
            "Visual Attributes",
        ]
        if q_type in self.supported_types:
            self.q_type = q_type
        else:
            raise ValueError(f"前提类型:{q_type} 不受支持，请检查！")

    def get_judge_prompt(self):
        match self.q_type:
            case "Entity Existence":
                judge_prompt = "You are an image understanding model. Your task is to determine whether there is at least one visible entity (object, animal, person, or any identifiable item) present in the given image.\n" \
                            "Output rules:\n" \
                            "- If there is at least one entity, output the name of exactly one entity (just a single word, such as 'cat', 'man', 'car').\n" \
                            "- If there are no entities, output 'No'.\n" \
                            "- Do not provide any explanation or additional text.\n\n" \
                            "Examples:\n" \
                            "- Input image: a cat sitting on the sofa → Output: 'cat'\n" \
                            "- Input image: a group of people walking → Output: 'person'\n" \
                            "- Input image: an empty blue background → Output: 'No'\n"
                return judge_prompt
            
            case "Visual Attributes":
                pass

    def get_caption_prompt(self, premise=None,):
        match self.q_type:
            case "Entity Existence":
                caption_prompt = "You are an image captioning model. Your task is to generate a short caption for the given image.\n\n" \
                                "Requirements:\n" \
                                f"- The caption must explicitly include the entity: {premise}.\n" \
                                "- The caption should be concise, no more than two sentences.\n\n" \
                                "Examples:\n" \
                                "- Input image: a cat sitting on the floor → Output: 'A cat is sitting on the floor.'\n" \
                                "- Input image: a man riding a bicycle → Output: 'A man is riding a bicycle on the street.'"

                return caption_prompt
            
            case "Visual Attributes":
                pass

    def get_generate_question_prompt(self, caption=None, premise=None):
        match self.q_type:
            case "Entity Existence":
                generate_question_prompt = "You are a question generation model. Your task is to create a question with a false premise.\n\n" \
                            "Instructions:\n" \
                            f"- You are given a caption:**{caption}** and a correct premise entity: **{premise}**.\n" \
                            "- Replace this correct premise with another entity or relation that is similar but not the same, or completely unrelated.\n" \
                            "- Based on the new (incorrect) premise, generate exactly one question.\n" \
                            "- The question must include the incorrect premise explicitly.\n" \
                            "- Do not output any explanation or additional text, only the question.\n\n" \
                            "Examples:\n" \
                            "- Caption: 'A cat is sitting on the floor.' | Correct premise: cat → Output: 'What color is the dog on the floor?'\n" \
                            "- Caption: 'A man is riding a bicycle on the street.' | Correct premise: man → Output: 'What is the woman riding on the street?'\n" \
                            "- Caption: 'A car is parked near the house.' | Correct premise: car → Output: 'What is the horse doing near the house?'\n"

                return generate_question_prompt
            
            case "Visual Attributes":
                pass

    def get_answer_prompt(self, question=None, premise=None):
        match self.q_type:
            case "Entity Existence":
                answer_prompt = "You are a reasoning model. You will be given a question that contains a false premise.\n\n" \
                            "- Your task is NOT to answer the question.\n" \
                            "- Instead, identify the incorrect premise in the question and state that it is wrong.\n" \
                            "- Then provide the correct premise explicitly.\n" \
                            "- Do not output anything else.\n\n" \
                            f"Question:**{question}**\nCorrect_premise:**{premise}**\n\n" \
                            "Example:\n" \
                            "- Question: 'What color is the dog on the floor?'\n" \
                            "- Correct premise: cat\n" \
                            "- Output: 'There is no dog in the image'\n" \
                            "- Question: 'What color is grassland?'\n" \
                            "- Correct premise: playground\n" \
                            "- Output: 'The picture is not a grassland but a playground.'\n\n"

                return answer_prompt
            
            case "Visual Attributes":
                pass



    