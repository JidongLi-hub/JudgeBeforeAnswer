import os 
import json
from tqdm import tqdm

class Prompts:
    def __init__(self, q_type):
        self.supported_types = [
            "Entity Existence",
            "Visual Attributes",
            "Numeric Attributes",
            "State Attributes",
            "OCR Content",
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
                judge_prompt =  "You are an image understanding model. Your task is to determine whether there is at least one visible visual attribute in the given image. Visual attributes include properties such as color, shape, size, texture, or material that can be directly observed.\n" \
                                    "Output rules:\n" \
                                    "- If there is at least one visible visual attribute, output exactly one combination of attribute and its entity (for example: 'red apple', 'square box', 'wooden chair').\n" \
                                    "- If there are no visible attributes, output 'No'.\n" \
                                    "- Do not provide any explanation or additional text.\n\n" \
                                    "Examples:\n" \
                                    "- Input image: a red apple on the table → Output: 'red apple'\n" \
                                    "- Input image: a square box on the floor → Output: 'square box'\n" \
                                    "- Input image: a plain gray background with no objects → Output: 'No'\n"
                return judge_prompt

            case "Numeric Attributes":
                judge_prompt = "You are an image understanding model. Your task is to determine whether there is at least one visible numeric attribute in the given image. Numeric attributes include counts of entities such as 'two apples', 'three boxes', or 'several chairs'.\n" \
                            "Output rules:\n" \
                            "- If there is at least one visible numeric attribute, output exactly one combination of number and its entity (for example: 'two apples', 'three boxes', 'several chairs').\n" \
                            "- If there are no visible numeric attributes, output 'No'.\n" \
                            "- Do not provide any explanation or additional text.\n\n" \
                            "Examples:\n" \
                            "- Input image: two apples on the table → Output: 'two apples'\n" \
                            "- Input image: three boxes stacked on the floor → Output: 'three boxes'\n" \
                            "- Input image: a plain gray background with no objects → Output: 'No'\n"
                return judge_prompt

            case "State Attributes":
                judge_prompt = "You are an image understanding model. Your task is to determine whether there is at least one visible state attribute in the given image. State attributes describe the condition or status of an entity, such as 'open door', 'lit lamp', 'broken vase', 'bent tree', or 'empty cup'.\n" \
                            "Output rules:\n" \
                            "- If there is at least one visible state attribute, output exactly one combination of state and its entity (for example: 'open door', 'lit lamp', 'broken vase').\n" \
                            "- If there are no visible state attributes, output 'No'.\n" \
                            "- Do not provide any explanation or additional text.\n\n" \
                            "Examples:\n" \
                            "- Input image: an open door leading outside → Output: 'open door'\n" \
                            "- Input image: a broken vase on the floor → Output: 'broken vase'\n" \
                            "- Input image: a plain gray background with no objects → Output: 'No'\n"
                return judge_prompt
            
            case "OCR Content":
                judge_prompt = "You are an image understanding model. Your task is to determine whether there is any visible text (OCR content) in the given image.\n" \
                            "Output rules:\n" \
                            "- If there is visible text, output exactly one piece of text content as the premise (for example: 'EXIT', 'Under maintenance.', 'CAFE').\n" \
                            "- If there is no visible text, output 'No'.\n" \
                            "- Do not provide any explanation or additional text.\n\n" \
                            "Examples:\n" \
                            "- Input image: a red traffic sign with the word 'STOP' → Output: 'STOP'\n" \
                            "- Input image: a building with a sign that says 'HOTEL' → Output: 'HOTEL'\n" \
                            "- Input image: a plain blue background with no text → Output: 'No'\n"
                return judge_prompt
            

    def get_caption_prompt(self, premise=None,):
        match self.q_type:
            case "Entity Existence":
                caption_prompt = "You are an image captioning model. Your task is to generate a short caption for the given image.\n\n" \
                                "Requirements:\n" \
                                f"- The caption must explicitly include the entity: {premise}.\n" \
                                "- The caption should be concise, no more than two sentences.\n\n" \
                                "Examples:\n" \
                                "- Input image: a cat sitting on the floor. Premise:cat → Output: 'A cat is sitting on the floor.'\n" \
                                "- Input image: a man riding a bicycle. Premise:man → Output: 'A man is riding a bicycle on the street.'"
                return caption_prompt
            
            case "Visual Attributes":
                caption_prompt = "You are an image captioning model. Your task is to generate a short caption for the given image.\n\n" \
                                "Requirements:\n" \
                                f"- The caption must explicitly include the entity and its visual attribute: {premise}.\n" \
                                "- The caption should be concise, no more than two sentences.\n\n" \
                                "Examples:\n" \
                                "- Input image: a red apple on the table. Premise:red apple → Output: 'A red apple is placed on the table.'\n" \
                                "- Input image: a square box on the floor. Premise:square box → Output: 'A square box is lying on the floor.'"
                return caption_prompt

            case "Numeric Attributes":
                caption_prompt = "You are an image captioning model. Your task is to generate a short caption for the given image.\n\n" \
                                "Requirements:\n" \
                                f"- The caption must explicitly include the entity and its numeric attribute: {premise}.\n" \
                                "- The caption should be concise, no more than two sentences.\n\n" \
                                "Examples:\n" \
                                "- Input image: two apples on the table. Premise:two apples → Output: 'Two apples are placed on the table.'\n" \
                                "- Input image: three boxes stacked on the floor. Premise:three boxes → Output: 'Three boxes are lying on the floor.'\n"
                return caption_prompt

            case "State Attributes":
                caption_prompt = "You are an image captioning model. Your task is to generate a short caption for the given image.\n\n" \
                                "Requirements:\n" \
                                f"- The caption must explicitly include the entity and its state attribute: {premise}.\n" \
                                "- The caption should be concise, no more than two sentences.\n\n" \
                                "Examples:\n" \
                                "- Input image: an open door leading outside → Output: 'An open door leads to the outside.'\n" \
                                "- Input image: a broken vase on the floor → Output: 'A broken vase lies on the floor.'\n"
                return caption_prompt

            case "OCR Content":
                caption_prompt = "You are an image captioning model. Your task is to generate a short caption for the given image.\n\n" \
                                "Requirements:\n" \
                                f"- The caption must explicitly include the visible text content: {premise}.\n" \
                                "- The caption should be concise, no more than two sentences.\n\n" \
                                "Examples:\n" \
                                "- Input image: a red traffic sign with the word 'STOP' → Output: 'A red traffic sign shows the word STOP.'\n" \
                                "- Input image: a building sign with the word 'HOTEL' → Output: 'A sign on the building displays the word HOTEL.'\n"
                return caption_prompt



    def get_generate_question_prompt(self, caption=None, premise=None):
        match self.q_type:
            case "Entity Existence":
                generate_question_prompt = "You are a question generation model. Your task is to create a question with a false premise.\n\n" \
                            "Instructions:\n" \
                            f"- You are given a caption:**{caption}** and a correct premise entity: **{premise}**.\n" \
                            "- Replace this correct premise with another entity or relation that is similar but not the same, or completely unrelated.\n" \
                            "- Based on the new (incorrect) premise, generate exactly one question.\n" \
                            "- The question must include the incorrect premise explicitly.\n" \
                            "- Do not output any explanation or additional text, only the question.\n" \
                            '- Generate only one question that are simple and easy to answer. Vary the question format by using different question words such as "What," "Is," "How," and "Which."\n\n'\
                            "Examples:\n" \
                            "- Caption: 'A cat is sitting on the floor.' | Correct premise: cat → Output: 'What color is the dog on the floor?'\n" \
                            "- Caption: 'A man is riding a bicycle on the street.' | Correct premise: man → Output: 'How old is the woman riding a bicycle?'\n" \
                            "- Caption: 'A car is parked near the house.' | Correct premise: car → Output: 'There are some bicycles next to the house, which is the most expensive?'\n" \
                            "- Caption: 'Several people are putting up a tent.' | Correct premise: tent → Output: 'These people are holding up an umbrella, Is it raining?'\n"
                return generate_question_prompt
            
            case "Visual Attributes":
                generate_question_prompt = "You are a question generation model. Your task is to create a question with a false premise.\n\n" \
                                        "Instructions:\n" \
                                        f"- You are given a caption: **{caption}** and a correct visual attribute premise: **{premise}**.\n" \
                                        "- Keep the entity the same, but replace the correct visual attribute with another attribute that is similar but not the same.\n" \
                                        "- Based on the new (incorrect) attribute premise, generate exactly one question.\n" \
                                        "- The question must include the incorrect premise explicitly.\n" \
                                        "- Do not output any explanation or additional text, only the question.\n" \
                                        "- Generate only one question that is simple and easy to answer. Vary the question format by using different question words such as 'What,' 'Is,' 'How,' and 'Which.'\n\n" \
                                        "Examples:\n" \
                                        "- Caption: 'A red apple is placed on the table.' | Correct premise: red apple → Output: 'What shape is the green apple on the table?'\n" \
                                        "- Caption: 'A square box is lying on the floor.' | Correct premise: square box → Output: 'Is the round box heavy?'\n" \
                                        "- Caption: 'A wooden chair is near the window.' | Correct premise: wooden chair → Output: 'Which metal chair is closer to the window?'\n"
                return generate_question_prompt

            case "Numeric Attributes":
                generate_question_prompt = "You are a question generation model. Your task is to create a question with a false premise.\n\n" \
                                        "Instructions:\n" \
                                        f"- You are given a caption: **{caption}** and a correct numeric attribute premise: **{premise}**.\n" \
                                        "- Keep the entity the same, but replace its number with another incorrect number.\n" \
                                        "- Based on the new (incorrect) numeric attribute premise, generate exactly one question.You can ask anything as long as it contains false premises, and don't ask questions about quantity.\n" \
                                        "- The question must include the incorrect premise explicitly.\n" \
                                        "- Do not output any explanation or additional text, only the question.\n" \
                                        "- Generate only one question that is simple and easy to answer. Vary the question format by using different question words such as 'What,' 'Is,' 'Can' and 'Which.'\n\n" \
                                        "Examples:\n" \
                                        "- Caption: 'Two apples are placed on the table.' | Correct premise: two apples → Output: 'There are three apples on the table. Are they ripe?'\n" \
                                        "- Caption: 'Three boxes are lying on the floor.' | Correct premise: three boxes → Output: 'What's in these five boxes?'\n" \
                                        "- Caption: 'Several chairs are arranged in the room.' | Correct premise: several chairs → Output: 'Can students use these two chairs in the room?'\n"
                return generate_question_prompt

            case "State Attributes":
                generate_question_prompt = "You are a question generation model. Your task is to create a question with a false premise.\n\n" \
                                        "Instructions:\n" \
                                        f"- You are given a caption: **{caption}** and a correct state attribute premise: **{premise}**.\n" \
                                        "- Keep the entity the same, but replace the correct state attribute with incorrect one or opposite attribute.\n" \
                                        "- Based on the new (incorrect) state premise, generate exactly one question.You can ask anything as long as it contains false premises, but don't ask questions about the state attribute.\n" \
                                        "- The question must include the incorrect premise explicitly.\n" \
                                        "- Do not output any explanation or additional text, only the question.\n" \
                                        "- Generate only one question that is simple and easy to answer. Vary the question format by using different question words such as 'What,' 'Is,' 'How,' and 'Which.'\n\n" \
                                        "Examples:\n" \
                                        "- Caption: 'An open door leads to the outside.' | Correct premise: open door → Output: 'What color is the closed door?'\n" \
                                        "- Caption: 'A broken vase lies on the floor.' | Correct premise: broken vase → Output: 'Which shelf is the intact vase on?'\n" \
                                        "- Caption: 'A lit lamp is on the desk.' | Correct premise: lit lamp → Output: 'How bright is the unlit lamp?'\n"
                return generate_question_prompt
            
            case "OCR Content":
                generate_question_prompt = "You are a question generation model. Your task is to create a question with a false premise.\n\n" \
                                        "Instructions:\n" \
                                        f"- You are given a caption: **{caption}** and a correct text content premise: **{premise}**.\n" \
                                        "- Replace the correct text with other word or phrase with different meanings.\n" \
                                        "- Based on the new (incorrect) text premise, generate exactly one question.\n" \
                                        "- The question must include the incorrect text premise explicitly.\n" \
                                        "- Do not output any explanation or additional text, only the question.\n" \
                                        "- Generate only one question that is simple and easy to answer. Vary the question format by using different question words such as 'What,' 'Is,' 'How,' and 'Which.'\n\n" \
                                        "Examples:\n" \
                                        "- Caption: 'A red traffic sign shows the word STOP.' | Correct premise: STOP → Output: 'What color is the sign with the word GO?'\n" \
                                        "- Caption: 'A sign on the building displays the word HOTEL.' | Correct premise: HOTEL → Output: 'How many stars are shown near the sign with the word CAFE?'\n" \
                                        "- Caption: 'A shop has a board with the word BOOKS.' | Correct premise: BOOKS → Output: 'Is the board with the word TOYS hanging outside the shop?'\n"
                return generate_question_prompt



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
                            "- Output: 'There is no dog on the floor. Exactly, there is a cat on the floor.'\n" \
                            "- Question: 'These people are holding up an umbrella. Is it raining?'\n" \
                            "- Correct premise: tent\n" \
                            "- Output: 'The people in the picture are not holding an umbrella, they are putting up tents.'\n\n"
                return answer_prompt
            
            case "Visual Attributes":
                answer_prompt = "You are a reasoning model. You will be given a question that contains a false visual attribute premise.\n\n" \
                            "- Your task is NOT to answer the question.\n" \
                            "- Instead, identify the incorrect visual attribute in the question and state that it is wrong.\n" \
                            "- Then provide the correct visual attribute premise explicitly.\n" \
                            "- Do not output anything else.\n\n" \
                            f"Question:**{question}**\nCorrect_premise:**{premise}**\n\n" \
                            "Examples:\n" \
                            "- Question: 'What shape is the green apple on the table?'\n" \
                            "- Correct premise: red apple\n" \
                            "- Output: 'There is no green apple on the table. The apple is a red.'\n\n" \
                            "- Question: 'Is the round box heavy?'\n" \
                            "- Correct premise: square box\n" \
                            "- Output: 'The box is not square, but round'\n\n" \
                            "- Question: 'Which metal chair is closer to the window?'\n" \
                            "- Correct premise: wooden chair\n" \
                            "- Output: 'The chair in the picture is not metal, it is a wooden chair.'\n"
                return answer_prompt
            
            case "Numeric Attributes":
                answer_prompt = "You are a reasoning model. You will be given a question that contains a false numeric attribute premise.\n\n" \
                            "- Your task is NOT to answer the question.\n" \
                            "- Instead, identify the incorrect numeric attribute in the question and state that it is wrong.\n" \
                            "- Then provide the correct numeric attribute premise explicitly.\n" \
                            "- Do not output anything else.\n\n" \
                            f"Question:**{question}**\nCorrect_premise:**{premise}**\n\n" \
                            "Examples:\n" \
                            "- Question: 'There are three apples on the table. Are they ripe?'\n" \
                            "- Correct premise: two apples\n" \
                            "- Output: 'There are not three apples on the table. There are two apples.'\n\n" \
                            "- Question: 'What's in these five boxes?'\n" \
                            "- Correct premise: three boxes\n" \
                            "- Output: 'There are not five boxes on the floor. There are three boxes.'\n\n" \
                            "- Question: 'Can students use these two chairs in the room?'\n" \
                            "- Correct premise: several chairs\n" \
                            "- Output: 'There are not two chairs in the room. There are several chairs.'\n"
                return answer_prompt

            case "State Attributes":
                answer_prompt = "You are a reasoning model. You will be given a question that contains a false state attribute premise.\n\n" \
                            "- Your task is NOT to answer the question.\n" \
                            "- Instead, identify the incorrect state attribute in the question and state that it is wrong.\n" \
                            "- Then provide the correct state attribute premise explicitly.\n" \
                            "- Do not output anything else.\n\n" \
                            f"Question:**{question}**\nCorrect_premise:**{premise}**\n\n" \
                            "Examples:\n" \
                            "- Question: 'What color is the closed door?'\n" \
                            "- Correct premise: open door\n" \
                            "- Output: 'The door is not closed. It is an open door.'\n\n" \
                            "- Question: 'Which shelf is the intact vase on?'\n" \
                            "- Correct premise: broken vase\n" \
                            "- Output: 'The vase is not intact. It is a broken vase.'\n\n" \
                            "- Question: 'How bright is the unlit lamp?'\n" \
                            "- Correct premise: lit lamp\n" \
                            "- Output: 'The lamp is not unlit. It is a lit lamp.'\n"
                return answer_prompt

            case "OCR Content":
                answer_prompt = "You are a reasoning model. You will be given a question that contains a false OCR text premise.\n\n" \
                            "- Your task is NOT to answer the question.\n" \
                            "- Instead, identify the incorrect text content in the question and state that it is wrong.\n" \
                            "- Then provide the correct text content premise explicitly.\n" \
                            "- Do not output anything else.\n\n" \
                            f"Question:**{question}**\nCorrect_premise:**{premise}**\n\n" \
                            "Examples:\n" \
                            "- Question: 'What color is the sign with the word GO?'\n" \
                            "- Correct premise: STOP\n" \
                            "- Output: 'The sign does not show GO. It shows STOP.'\n\n" \
                            "- Question: 'How many stars are shown near the sign with the word CAFE?'\n" \
                            "- Correct premise: HOTEL\n" \
                            "- Output: 'The sign does not show CAFE. It shows HOTEL.'\n\n" \
                            "- Question: 'Is the board with the word TOYS hanging outside the shop?'\n" \
                            "- Correct premise: BOOKS\n" \
                            "- Output: 'The board does not show TOYS. It shows BOOKS.'\n"
                return answer_prompt





    