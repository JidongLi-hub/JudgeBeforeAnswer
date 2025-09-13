import os 
import json
from tqdm import tqdm

class Prompts:
    supported_types = [
            "Entity Existence",
            "Visual Attributes",
            "Numeric Attributes",
            "State Attributes",
            "OCR Content",
            "Symbol Meaning",
            "Spatial Relation",
            "Interaction Relation",
            "Possessive Relation",
            "Emotion",
            "Scene",
            "Logical",
            "Commonsense"
        ]
    def __init__(self, q_type):
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
            
            case "Symbol Meaning":
                judge_prompt = "You are an image understanding model. Your task is to determine whether there is at least one visible symbol or sign in the given image. Symbols include traffic signs, icons, logos, or other graphical signs with specific meanings.\n" \
                                "Output rules:\n" \
                                "- If there is at least one visible symbol, output the name of exactly one symbol and its meaning (for example: 'STOP sign', 'No Parking sign', 'Right Turn arrow').\n" \
                                "- If there are no visible symbols, output 'No'.\n" \
                                "- Do not provide any explanation or additional text.\n\n" \
                                "Examples:\n" \
                                "- Input image: a red octagon traffic sign with the word STOP → Output: 'STOP sign'\n" \
                                "- Input image: a blue circular sign with a right turn arrow → Output: 'Right Turn sign'\n" \
                                "- Input image: a plain green background with no signs → Output: 'No'\n"
                return judge_prompt
            
            case "Spatial Relation":
                judge_prompt = "You are an image understanding model. Your task is to determine whether there is at least one visible spatial relation in the given image.\n" \
                            "A spatial relation must include three elements: subject, relation, and object (for example: 'apple on table', 'river left of tree', 'cake next to box').\n" \
                            "Output rules:\n" \
                            "- If there is at least one spatial relation, output exactly one as a short phrase in the form: 'subject + relation + object'.\n" \
                            "- If there are no visible spatial relations, output 'No'.\n" \
                            "- Do not provide any explanation or additional text.\n\n" \
                            "Examples:\n" \
                            "- Input image: an apple placed on the table → Output: 'apple on the table'\n" \
                            "- Input image: a river flowing to the left of a tree → Output: 'river left of tree'\n" \
                            "- Input image: an empty plain background → Output: 'No'\n"
                return judge_prompt

            case "Interaction Relation":
                judge_prompt = "You are an image understanding model. Your task is to determine whether there is at least one visible interaction relation in the given image.\n" \
                                "An interaction relation must include three elements: subject (Agent), action, and object (Patient) (for example: 'person is holding horse', 'girl is chasing pigeon').\n" \
                                "Output rules:\n" \
                                "- If there is at least one interaction relation, output exactly one as a short phrase in the form: 'subject + action + object'.\n" \
                                "- If there are no visible interaction relations, output 'No'.\n" \
                                "- Do not provide any explanation or additional text.\n\n" \
                                "Examples:\n" \
                                "- Input image: a person holding a horse → Output: 'person is holding horse'\n" \
                                "- Input image: a girl chasing pigeons → Output: 'girl is chasing pigeon'\n" \
                                "- Input image: a dog sitting alone → Output: 'No'\n"
                return judge_prompt

            case "Possessive Relation":
                judge_prompt =  "You are an image understanding model. Your task is to determine whether the given image contains a visible possessive relation. " \
                            "Possessive relations describe how one entity is attached to, composed of, or dependent on another entity (for example: 'bricks build castle', 'rope ties dog', 'horse pulls cart').\n\n" \
                            "Output rules:\n" \
                            "- If there is at least one possessive relation, output exactly one combination of subject, relation, and object (for example: 'bricks build castle', 'rope ties dog').\n" \
                            "- If there is no visible possessive relation, output 'No'.\n" \
                            "- Do not provide any explanation or extra text.\n\n" \
                            "Examples:\n" \
                            "- Input image: a horse pulling a cart → Output: 'horse pulls cart'\n" \
                            "- Input image: a rope tied to a dog → Output: 'rope ties dog'\n" \
                            "- Input image: just a plain gray wall → Output: 'No'\n"
                return judge_prompt

            case "Emotion":
                judge_prompt = "You are an image understanding model. Your task is to determine whether there is at least one visible emotional state in the given image.\n" \
                            "An emotional state must include two elements: subject (human or animal) and the expressed emotion (for example: 'man is sad', 'woman is joyful', 'dog is fearful').\n" \
                            "Output rules:\n" \
                            "- If there is at least one emotional state, output exactly one as a short phrase.\n" \
                            "- If there are no visible emotional states, output 'No'.\n" \
                            "- Do not provide any explanation or additional text.\n\n" \
                            "Examples:\n" \
                            "- Input image: a man crying alone → Output: 'man is sad'\n" \
                            "- Input image: a woman smiling with excitement → Output: 'woman is joyful'\n" \
                            "- Input image: Three dogs trembling in a storm → Output: 'dogs are fearful'\n" \
                            "- Input image: an empty plain background → Output: 'No'\n"
                return judge_prompt
            
            case "Scene":
                judge_prompt = "You are an image understanding model. Your task is to determine whether there is at least one visible scene activity or environmental state in the given image.\n" \
                            "A scene premise must include a subject (human, group, or environment) and the described action or state (for example: 'farmer is planting crops', 'audience is entering', 'sky is dark and cloudy').\n" \
                            "Output rules:\n" \
                            "- If there is at least one scene premise, output exactly one as a short phrase\n" \
                            "- If there are no visible scene premises, output 'No'.\n" \
                            "- Do not provide any explanation or additional text.\n\n" \
                            "Examples:\n" \
                            "- Input image: a farmer working in the field → Output: 'farmer is planting crops'\n" \
                            "- Input image: a crowd going into a stadium → Output: 'audience is entering'\n" \
                            "- Input image: dark clouds covering the sky → Output: 'sky is dark and cloudy'\n" \
                            "- Input image: an empty plain background → Output: 'No'\n"
                return judge_prompt
            
            case "Logical":
                judge_prompt = "You are an image-text understanding model. Your task is to determine whether there is at least one logical sequence or causal relation expressed in the given image description.\n" \
                            "A logical premise must describe a cause-effect or temporal sequence (for example: 'ice cream fell on the ground, then it melted', 'the weather is hot, so people use umbrellas').\n" \
                            "Output rules:\n" \
                            "- If there is at least one logical sequence, output exactly one as a short phrase in the form: 'event1 → event2'.\n" \
                            "- If there are no visible logical sequences, output 'No'.\n" \
                            "- Do not provide any explanation or additional text.\n\n" \
                            "Examples:\n" \
                            "- Input image: ice cream on the ground melting → Output: 'ice cream fell → it melted'\n" \
                            "- Input image: people holding umbrellas under the sun → Output: 'weather is hot → people use umbrellas'\n" \
                            "- Input image: a man walking without any related consequence → Output: 'No'\n"
                return judge_prompt
            
            case "Commonsense":
                judge_prompt = "You are an image understanding model. Your task is to determine whether there is at least one commonsense fact visible in the given image.\n" \
                            "A commonsense fact refers to something that normally happens or exists in the real world (for example: 'apple falls from a tree to the ground', 'boat floats on water', 'sun rises in the east').\n" \
                            "Output rules:\n" \
                            "- If there is at least one commonsense fact, output exactly one as a short phrase.\n" \
                            "- If there are no visible commonsense facts, output 'No'.\n" \
                            "- Do not provide any explanation or additional text.\n\n" \
                            "Examples:\n" \
                            "- Input image: apples falling from a tree → Output: 'apple falls to ground'\n" \
                            "- Input image: a boat moving on the water → Output: 'boat floats on water'\n" \
                            "- Input image: the sun appearing at sunrise → Output: 'sun rises in east'\n" \
                            "- Input image: an abstract background with no commonsense element → Output: 'No'\n"
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
                                "- Input image: an open door leading outside. Premise: open door → Output: 'An open door leads to the outside.'\n" \
                                "- Input image: a broken vase on the floor. Premise:broken vase → Output: 'A broken vase lies on the floor.'\n"
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

            case "Symbol Meaning":
                caption_prompt = "You are an image captioning model. Your task is to generate a short caption for the given image.\n\n" \
                                "Requirements:\n" \
                                f"- The caption must explicitly include the visible symbol: {premise}.\n" \
                                "- The caption should be concise, no more than two sentences.\n\n" \
                                "Examples:\n" \
                                "- Input image: a red octagon traffic sign with the word STOP → Output: 'A STOP sign is displayed at the street corner.'\n" \
                                "- Input image: a blue circular sign with a right turn arrow → Output: 'A Right Turn sign is posted on the road.'\n"
                return caption_prompt
            
            case "Spatial Relation":
                caption_prompt = "You are an image captioning model. Your task is to generate a short caption for the given image.\n\n" \
                                "Requirements:\n" \
                                f"- The caption must explicitly include the spatial relation: {premise}.\n" \
                                "- The caption should be concise, no more than two sentences.\n\n" \
                                "Examples:\n" \
                                "- Input image: an apple placed on the table. Premise:apple on the table → Output: 'An apple is resting on the table.'\n" \
                                "- Input image: a river flowing to the left of a tree. Premise:river left of the tree → Output: 'A river runs on the left side of the tree.'\n"
                return caption_prompt
            
            case "Interaction Relation":
                caption_prompt = "You are an image captioning model. Your task is to generate a short caption for the given image.\n\n" \
                                "Requirements:\n" \
                                f"- The caption must explicitly include the interaction relation premise: {premise}.\n" \
                                "- The caption should be concise, no more than two sentences.\n\n" \
                                "Examples:\n" \
                                "- Input image: a person holding a horse. Premise: person is holding horse → Output: 'A person is holding the horse firmly.'\n" \
                                "- Input image: a girl chasing pigeons. Premise: girl is chasing pigeon → Output: 'The girl is running after the pigeons in the park.'\n"
                return caption_prompt

            
            case "Possessive Relation":
                caption_prompt =  "You are an image captioning model. Your task is to generate a short caption for the given image.\n\n" \
                                "Requirements:\n" \
                                f"- The caption must explicitly include the possessive relation: {premise}.\n" \
                                "- The caption should be concise, no more than two sentences.\n\n" \
                                "Examples:\n" \
                                "- Input image: A horse pulled the cart and walked on the road. Premise: 'horse pulls cart' → Output: 'A horse is pulling a cart down the road.'\n" \
                                "- Input image: Some bricks build a large beautiful castle. Premise: 'bricks build castle' → Output: 'A large castle made of stacked bricks.'\n"
                return caption_prompt

            case "Emotion":
                caption_prompt = "You are an image captioning model. Your task is to generate a short caption for the given image.\n\n" \
                                "Requirements:\n" \
                                f"- The caption must explicitly include the emotional state premise: {premise}.\n" \
                                "- The caption should be concise, no more than two sentences.\n\n" \
                                "Examples:\n" \
                                "- Input image: a man crying alone. Premise: man is sad → Output: 'A man is sitting by himself, looking very sad.'\n" \
                                "- Input image: a woman smiling with excitement. Premise: woman is joyful → Output: 'The woman is clearly joyful as she laughs with happiness.'\n" \
                                "- Input image: a dog trembling in a storm. Premise: dog is fearful → Output: 'The dog appears fearful in the heavy rain.'\n"
                return caption_prompt

            case "Scene":
                caption_prompt = "You are an image captioning model. Your task is to generate a short caption for the given image.\n\n" \
                                "Requirements:\n" \
                                f"- The caption must explicitly include the scene premise: {premise}.\n" \
                                "- The caption should be concise, no more than two sentences.\n\n" \
                                "Examples:\n" \
                                "- Input image: a farmer working in the field. Premise: farmer is planting crops → Output: 'A farmer is planting crops in the open field.'\n" \
                                "- Input image: a crowd going into a stadium. Premise: audience is entering → Output: 'The audience is entering the stadium for the event.'\n" \
                                "- Input image: dark clouds covering the sky. Premise: sky is dark and cloudy → Output: 'The sky is dark and cloudy as if a storm is approaching.'\n"
                return caption_prompt
            
            case "Logical":
                caption_prompt = "You are an image captioning model. Your task is to generate a short caption for the given image.\n\n" \
                                "Requirements:\n" \
                                f"- The caption must explicitly include the logical sequence premise: {premise}.\n" \
                                "- The caption should be concise, no more than two sentences.\n\n" \
                                "Examples:\n" \
                                "- Input image: ice cream on the ground melting. Premise: ice cream fell → it melted → Output: 'The ice cream fell on the ground, and then it melted quickly.'\n" \
                                "- Input image: people holding umbrellas under the sun. Premise: weather is hot → people use umbrellas → Output: 'The weather is very hot, so people are holding umbrellas.'\n"
                return caption_prompt
            
            case "Commonsense":
                caption_prompt = "You are an image captioning model. Your task is to generate a short caption for the given image.\n\n" \
                            "Requirements:\n" \
                            f"- The caption must explicitly include the commonsense premise: {premise}.\n" \
                            "- The caption should be concise, no more than two sentences.\n\n" \
                            "Examples:\n" \
                            "- Input image: apples falling from a tree. Premise: apple falls to ground → Output: 'The apple falls from the tree onto the ground.'\n" \
                            "- Input image: a boat moving on the water. Premise: boat floats on water → Output: 'A boat is floating on the calm water.'\n" \
                            "- Input image: the sun appearing at sunrise. Premise: sun rises in east → Output: 'The sun is rising in the east, casting a warm glow.'\n"
                return caption_prompt


    def get_generate_real_question_prompt(self, caption=None, premise=None):
        generate_real_question_prompt = "You are a question generation model. Your task is to create a question that includes the correct premise.\n\n" \
                                    "Instructions:\n" \
                                    f"- You are given a caption: **{caption}** and a correct premise: **{premise}**.\n" \
                                    "- The question must include the correct premise explicitly.\n" \
                                    "- The question should NOT directly ask about the premise itself. Instead, it should ask about another aspect of the scene while incorporating the correct premise.\n" \
                                    "- Don't ask subjective questions, just ask objective questions in caption, and they should be very easy to answer, and you can answer them in one or two sentences." \
                                    "- Do not output any explanation or additional text, only the question.\n" \
                                    "- Generate only one question that is simple, clear, and easy to answer. You can vary the question word, e.g., 'What,' 'How,' 'Which,' or 'Why.'\n\n" \
                                    "Examples:\n" \
                                    "- Caption: 'An apple is resting on the table.' | Correct premise: apple on table → Output: 'The apple is on the table, Can the little boy reach it?'\n" \
                                    "- Caption: 'A river runs on the left side of the tree.' | Correct premise: river left of tree → Output: 'The river is on the left of the tree, what animals might come to drink from it?'\n" \
                                    "- Caption: 'A cake is placed next to a box.' | Correct premise: cake next to box → Output: 'The cake is next to the box, Please describe the color composition of the cake.?'\n"

        return generate_real_question_prompt

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
                                        "- Based on the new (incorrect) numeric attribute premise, generate exactly one question.You can ask anything as long as it contains false premises, and don't ask question about quantity.\n" \
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
                                        "- Based on the new (incorrect) state premise, generate exactly one question.You can ask anything as long as it contains false premises, but don't ask question about the state attribute.\n" \
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
                                        "- Replace the correct text with similar or opposite text.\n" \
                                        "- Based on the new (incorrect) text premise, generate exactly one question.You can ask anything as long as it contains false premises, but don't ask question about the text premise itself.\n" \
                                        "- The question must include the incorrect text premise explicitly.\n" \
                                        "- Do not output any explanation or additional text, only the question.\n" \
                                        "- Generate only one question that is simple and easy to answer. Vary the question format by using different question words such as 'What,' 'Is,' 'How,' and 'Which.'\n\n" \
                                        "Examples:\n" \
                                        "- Caption: 'A red traffic sign shows the word STOP.' | Correct premise: STOP → Output: 'What color is the sign with the word GO?'\n" \
                                        "- Caption: 'A sign on the building displays the word HOTEL.' | Correct premise: HOTEL → Output: 'How many stars are shown near the sign with the word CAFE?'\n" \
                                        "- Caption: 'A shop has a board with the word BOOKS.' | Correct premise: BOOKS → Output: 'Is the board with the word TOYS hanging outside the shop?'\n"
                return generate_question_prompt
            
            case "Symbol Meaning":
                generate_question_prompt = "You are a question generation model. Your task is to create a question with a false premise.\n\n" \
                                    "Instructions:\n" \
                                    f"- You are given a caption: **{caption}** and a correct symbol premise: **{premise}**.\n" \
                                    "- Replace the correct symbol with another, different symbol meaning (for example, replace 'STOP sign' with 'No Parking sign').\n" \
                                    "- Based on the new (incorrect) symbol premise, generate exactly one question.You can ask anything as long as it contains false premises, but don't ask question about the symbol itself.\n" \
                                    "- The question must include the incorrect premise explicitly.\n" \
                                    "- Do not output any explanation or additional text, only the question.\n" \
                                    "- Generate only one question that is simple and easy to answer. Vary the question format by using different question words such as 'What,' 'Is,' 'How,' and 'Which.'\n\n" \
                                    "Examples:\n" \
                                    "- Caption: 'A STOP sign is displayed at the street corner.' | Correct premise: STOP sign → Output: 'What does the No Parking sign mean at the corner?'\n" \
                                    "- Caption: 'A Right Turn sign is posted on the road.' | Correct premise: Right Turn sign → Output: 'Is the Left Turn sign visible on the road?'\n" \
                                    "- Caption: 'A triangle warning sign is near the construction site.' | Correct premise: Warning sign → Output: 'Which direction does the Pedestrian Crossing sign point to?'\n"
                return generate_question_prompt
            
            case "Spatial Relation":
                generate_question_prompt = "You are a question generation model. Your task is to create a question with a false spatial relation premise.\n\n" \
                                        "Instructions:\n" \
                                        f"- You are given a caption: **{caption}** and a correct spatial relation premise: **{premise}**.\n" \
                                        "- Keep the subject and object the same, but replace the spatial relation with another relation that is incorrect (for example: replace 'on' with 'under', 'left of' with 'right of').\n" \
                                        "- Based on this incorrect relation, generate exactly one question that includes the incorrect premise.\n" \
                                        "- The question should not directly ask about the false relation itself (e.g., avoid 'Is the apple under the table?'). Instead, ask about another aspect of the scene as long as including the incorrect relation.\n" \
                                        "- Do not output any explanation or additional text, only the question.\n" \
                                        "- Generate only one question that is simple, clear, and easy to answer. Vary the question format by using different question words such as 'What,' 'How,' 'Which,' and 'Why.'\n\n" \
                                        "Examples:\n" \
                                        "- Caption: 'An apple is resting on the table.' | Correct premise: apple on table → Output: 'The apple is under the table, what might the child do to find it?'\n" \
                                        "- Caption: 'A river runs on the left side of the tree.' | Correct premise: river left of tree → Output: 'The river were on the right of the tree, which path would animals take to drink water?'\n" \
                                        "- Caption: 'A cake is placed next to a box.' | Correct premise: cake next to box → Output: 'The cake were behind the box, what would you see first when entering the room?'\n"
                return generate_question_prompt
            
            case "Interaction Relation":
                generate_question_prompt = "You are a question generation model. Your task is to create a question with a false interaction relation premise.\n\n" \
                                        "Instructions:\n" \
                                        f"- You are given a caption: **{caption}** and a correct interaction relation premise: **{premise}**.\n" \
                                        "- Keep the action the same, but swap the subject and object to make the relation incorrect (for example: replace 'person is holding horse' with 'horse is holding person', 'girl is chasing pigeon' with 'pigeon is chasing girl').\n" \
                                        "- Based on this incorrect premise, generate exactly one question that includes the false interaction relation.\n" \
                                        "- The question should not directly ask about the false relation itself (e.g., avoid 'Is the horse holding the person?'). Instead, ask about another aspect of the scene while still including the incorrect premise.\n" \
                                        "- Do not output any explanation or additional text, only the question.\n" \
                                        "- Generate only one question that is simple, clear, and easy to answer. Vary the question format by using different question words such as 'What,' 'How,' 'Which,' and 'Why.'\n\n" \
                                        "Examples:\n" \
                                        "- Caption: 'A person is holding the horse firmly.' | Correct premise: person is holding horse → Output: 'The horse is holding the person, where would they go?'\n" \
                                        "- Caption: 'The girl is running after the pigeons in the park.' | Correct premise: girl is chasing pigeon → Output: 'The pigeon is chasing the girl. Is there anyone around who can help? '\n"
                return generate_question_prompt


            case "Possessive Relation":  # 提问方式可能需要修改
                generate_question_prompt = f" You are a question generation model. Your task is to generate a natural question based on a given image caption and its correct premise.\n\n" \
                                        f"- You are given a caption: **{caption}** and a correct possessive relation premise: **{premise}**.\n" \
                                        "Steps:\n" \
                                        "1. Replace the correct possessive relation (premise) with an incorrect one by swapping subject and object.\n" \
                                        "2. Use the incorrect premise to generate a question. The incorrect premise must appear in the question as a declarative statement (do NOT start the question with 'If').\n" \
                                        "   - State the incorrect relation as a factual clause (e.g., 'The cart is pulling the horse.') and then ask a question about another aspect of the scene that depends on that clause.\n" \
                                        "   - The question should NOT directly ask whether the false relation is true (avoid 'Is the cart pulling the horse?').\n" \
                                        "   - The question may ask about actions, consequences, timings, or other entities in the scene, but must include the incorrect relation statement.\n" \
                                        "3. Generate exactly one concise question. Do not output any explanation or additional text.\n\n" \
                                        "Examples:\n" \
                                        "- Caption: 'A horse is pulling a cart down the road.'\n" \
                                        "  Correct premise: 'horse pulls cart'\n" \
                                        "  Generated Question: 'The cart is pulling the horse. How far could they travel together?'\n\n" \
                                        "- Caption: 'A large castle made of stacked bricks.'\n" \
                                        "  Correct premise: 'bricks build castle'\n" \
                                        "  Generated Question: 'The castle builds the bricks. How strong will the resulting structure be?'\n\n" \
                                        "- Caption: 'A rope is tied to a dog.'\n" \
                                        "  Correct premise: 'rope ties dog'\n" \
                                        "  Generated Question: 'The dog is tying the rope. What might happen next in the scene?'\n"
                return generate_question_prompt

            case "Emotion":
                generate_question_prompt = "You are a question generation model. Your task is to create a question with a false emotional state premise.\n\n" \
                                    "Instructions:\n" \
                                    f"- You are given a caption: **{caption}** and a correct emotional state premise: **{premise}**.\n" \
                                    "- Keep the subject the same, but replace the emotion with another emotion that is incorrect (for example: replace 'sad' with 'joyful', 'joyful' with 'angry', 'fearful' with 'calm').\n" \
                                    "- Based on this incorrect emotion, generate exactly one question that includes the incorrect premise.\n" \
                                    "- The question should not directly ask about the false emotion itself (e.g., avoid 'Is the man happy?'). Instead, ask about another aspect of the scene while still including the incorrect premise.\n" \
                                    "- Do not output any explanation or additional text, only the question.\n" \
                                    "- Generate only one question that is simple, clear, and easy to answer. Vary the question format by using different question words such as 'What,' 'How,' 'Which,' and 'Why.'\n\n" \
                                    "Examples:\n" \
                                    "- Caption: 'A man is sitting by himself, looking very sad.' | Correct premise: man is sad → Output: 'The man is joyful, what might he be celebrating?'\n" \
                                    "- Caption: 'The woman is clearly joyful as she laughs with happiness.' | Correct premise: woman is joyful → Output: 'The woman is angry, how would her friends try to calm her down?'\n" \
                                    "- Caption: 'The dog appears fearful in the heavy rain.' | Correct premise: dog is fearful → Output: 'The dog is calm, which game would it play with the children in this weather?'\n"
                return generate_question_prompt

            case "Scene":
                generate_question_prompt = "You are a question generation model. Your task is to create a question with a false scene premise.\n\n" \
                           "Instructions:\n" \
                           f"- You are given a caption: **{caption}** and a correct scene premise: **{premise}**.\n" \
                           "- Keep the subject the same, but replace the action/state with another that is incorrect (for example: replace 'planting crops' with 'harvesting', 'entering' with 'leaving', 'dark and cloudy' with 'bright and sunny').\n" \
                           "- Based on this incorrect premise, generate exactly one question that includes the incorrect premise.\n" \
                           "- The question should not directly ask about the false premise itself (e.g., avoid 'Is the farmer harvesting?'). Instead, ask about another aspect of the scene while still including the incorrect premise.\n" \
                           "- Do not output any explanation or additional text, only the question.\n" \
                           "- Generate only one question that is simple, clear, and easy to answer. Vary the question format by using different question words such as 'What,' 'How,' 'Which,' and 'Why.'\n\n" \
                           "Examples:\n" \
                           "- Caption: 'A farmer is planting crops in the open field.' | Correct premise: farmer is planting crops → Output: 'The farmer is harvesting the crops, what tools might he use?'\n" \
                           "- Caption: 'The audience is entering the stadium for the event.' | Correct premise: audience is entering → Output: 'The audience is leaving the stadium, which direction would the traffic flow?'\n" \
                           "- Caption: 'The sky is dark and cloudy as if a storm is approaching.' | Correct premise: sky is dark and cloudy → Output: 'The sky is bright and sunny, what outdoor activities might people enjoy?'\n"
                return generate_question_prompt
            
            case "Logical":
                generate_question_prompt = "You are a question generation model. Your task is to create a question with a false logical sequence premise.\n\n" \
                                    "Instructions:\n" \
                                    f"- You are given a caption: **{caption}** and a correct logical premise: **{premise}**.\n" \
                                    "- Keep the events the same, but reverse or alter the logical order so that the relation is incorrect (for example: replace 'ice cream fell → it melted' with 'ice cream melted → it fell', 'weather is hot → people use umbrellas' with 'people use umbrellas → weather is hot').\n" \
                                    "- Based on this incorrect premise, generate exactly one question that includes the incorrect logic.\n" \
                                    "- The question should not directly ask about the false logic itself (e.g., avoid 'Did the ice cream melt before falling?'). Instead, ask about another aspect of the scene while still including the incorrect premise.\n" \
                                    "- Do not output any explanation or additional text, only the question.\n" \
                                    "- Generate only one question that is simple, clear, and easy to answer. Vary the question format by using different question words such as 'What,' 'How,' 'Which,' and 'Why.'\n\n" \
                                    "Examples:\n" \
                                    "- Caption: 'The ice cream fell on the ground, and then it melted quickly.' | Correct premise: ice cream fell → it melted → Output: 'The ice cream melted first and then fell, what would a child do after seeing it?'\n" \
                                    "- Caption: 'The weather is very hot, so people are holding umbrellas.' | Correct premise: weather is hot → people use umbrellas → Output: 'People used umbrellas, so the weather became hot, how would this affect outdoor activities?'\n"
                return generate_question_prompt
            
            case "Commonsense":
                generate_question_prompt = "You are a question generation model. Your task is to create a question with a false commonsense premise.\n\n" \
                                    "Instructions:\n" \
                                    f"- You are given a caption: **{caption}** and a correct commonsense premise: **{premise}**.\n" \
                                    "- Keep the subject the same, but replace the commonsense action/state with an incorrect version that contradicts basic real-world knowledge (for example: replace 'apple falls to ground' with 'apple flies up to tree', 'boat floats on water' with 'boat moves on road').\n" \
                                    "- Based on this incorrect premise, generate exactly one question that includes the incorrect commonsense fact.\n" \
                                    "- The question should not directly ask about the false commonsense (e.g., avoid 'Does the apple fly up?'). Instead, ask about another aspect of the scene while still including the incorrect premise.\n" \
                                    "- Do not output any explanation or additional text, only the question.\n" \
                                    "- Generate only one question that is simple, clear, and easy to answer. Vary the question format by using different question words such as 'What,' 'How,' 'Which,' and 'Why.'\n\n" \
                                    "Examples:\n" \
                                    "- Caption: 'The apple falls from the tree onto the ground.' | Correct premise: apple falls to ground → Output: 'The apple is flying up to the tree, what would happen if a child tried to catch it?'\n" \
                                    "- Caption: 'A boat is floating on the calm water.' | Correct premise: boat floats on water → Output: 'The boat is driving on the road, which vehicles might it meet?'\n" \
                                    "- Caption: 'The sun is rising in the east, casting a warm glow.' | Correct premise: sun rises in east → Output: 'The sun is rising in the west, how would people adjust their morning routines?'\n"
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

            case "Symbol Meaning":
                answer_prompt = "You are a reasoning model. You will be given a question that contains a false symbol premise.\n\n" \
                            "- Your task is NOT to answer the question.\n" \
                            "- Instead, identify the incorrect symbol in the question and state that it is wrong.\n" \
                            "- Then provide the correct symbol premise explicitly.\n" \
                            "- Do not output anything else.\n\n" \
                            f"Question:**{question}**\nCorrect_premise:**{premise}**\n\n" \
                            "Examples:\n" \
                            "- Question: 'What does the No Parking sign mean at the corner?'\n" \
                            "- Correct premise: STOP sign\n" \
                            "- Output: 'There is no No Parking sign at the corner. It is a STOP sign.'\n\n" \
                            "- Question: 'Is the Left Turn sign visible on the road?'\n" \
                            "- Correct premise: Right Turn sign\n" \
                            "- Output: 'There is no Left Turn sign on the road. It is a Right Turn sign.'\n\n" \
                            "- Question: 'Which direction does the Pedestrian Crossing sign point to?'\n" \
                            "- Correct premise: Warning sign\n" \
                            "- Output: 'There is no Pedestrian Crossing sign. It is a Warning sign near the construction site.'\n"
                return answer_prompt
            
            case "Spatial Relation":
                answer_prompt = "You are a reasoning model. You will be given a question that contains a false spatial relation premise.\n\n" \
                            "- Your task is NOT to answer the question.\n" \
                            "- Instead, identify the incorrect spatial relation in the question and state that it is wrong.\n" \
                            "- Then provide the correct spatial relation premise explicitly.\n" \
                            "- Do not output anything else.\n\n" \
                            f"Question:**{question}**\nCorrect_premise:**{premise}**\n\n" \
                            "Examples:\n" \
                            "- Question: 'The apple is under the table, what might the child do to find it?'\n" \
                            "- Correct premise: apple on table\n" \
                            "- Output: 'The apple is not under the table. It is on the table.'\n\n" \
                            "- Question: 'The river were on the right of the tree, which path would animals take to drink water?'\n" \
                            "- Correct premise: river left of tree\n" \
                            "- Output: 'The river is not on the right of the tree. It is on the left of the tree.'\n\n" \
                            "- Question: 'The cake were behind the box, what would you see first when entering the room?'\n" \
                            "- Correct premise: cake next to box\n" \
                            "- Output: 'The cake is not behind the box. It is next to the box.'\n"
                return answer_prompt

            case "Interaction Relation":
                answer_prompt = "You are a reasoning model. You will be given a question that contains a false interaction relation premise.\n\n" \
                            "- Your task is NOT to answer the question.\n" \
                            "- Instead, identify the incorrect interaction relation in the question and state that it is wrong.\n" \
                            "- Then provide the correct interaction relation premise explicitly.\n" \
                            "- Do not output anything else.\n\n" \
                            f"Question:**{question}**\nCorrect_premise:**{premise}**\n\n" \
                            "Examples:\n" \
                            "- Question: 'The horse is holding the person, how would people react?'\n" \
                            "- Correct premise: person is holding horse\n" \
                            "- Output: 'The horse is not holding the person. The person is holding the horse.'\n\n" \
                            "- Question: 'The pigeon is chasing the girl, what might happen next?'\n" \
                            "- Correct premise: girl is chasing pigeon\n" \
                            "- Output: 'The pigeon is not chasing the girl. The girl is chasing the pigeon.'\n"
                return answer_prompt

            case "Possessive Relation":
                answer_prompt =  "You are a reasoning model. You will be given a question that contains a false possessive relation premise.\n\n" \
                                "- Your task is NOT to answer the question.\n" \
                                "- Instead, identify the incorrect possessive relation in the question and state that it is wrong.\n" \
                                "- Then provide the correct relation explicitly.\n" \
                                "- Do not output anything else.\n\n" \
                                f"Question:**{question}**\nCorrect_premise:**{premise}**\n\n" \
                                "Examples:\n" \
                                "- Question: 'If the cart is pulling the horse, how far could they travel together?'\n" \
                                "- Correct premise: horse pulls cart\n" \
                                "- Output: 'The cart is not pulling the horse. Exactly, the horse pulls the cart.'\n\n" \
                                "- Question: 'If the castle builds the bricks, how strong will the structure be?'\n" \
                                "- Correct premise: bricks build castle\n" \
                                "- Output: 'The castle does not build the bricks. Exactly, the bricks build the castle.'\n"
                return answer_prompt

            case "Emotion":
                answer_prompt = "You are a reasoning model. You will be given a question that contains a false emotional state premise.\n\n" \
                            "- Your task is NOT to answer the question.\n" \
                            "- Instead, identify the incorrect emotional state in the question and state that it is wrong.\n" \
                            "- Then provide the correct emotional state premise explicitly.\n" \
                            "- Do not output anything else.\n\n" \
                            f"Question:**{question}**\nCorrect_premise:**{premise}**\n\n" \
                            "Examples:\n" \
                            "- Question: 'The man is joyful, what might he be celebrating?'\n" \
                            "- Correct premise: man is sad\n" \
                            "- Output: 'The man is not joyful. He is sad.'\n\n" \
                            "- Question: 'The woman is angry, how would her friends try to calm her down?'\n" \
                            "- Correct premise: woman is joyful\n" \
                            "- Output: 'The woman is not angry. She is joyful.'\n\n" \
                            "- Question: 'The dog is calm, which game would it play with the children in this weather?'\n" \
                            "- Correct premise: dog is fearful\n" \
                            "- Output: 'The dog is not calm. It is fearful.'\n"
                return answer_prompt
            
            case "Scene":
                answer_prompt = "You are a reasoning model. You will be given a question that contains a false scene premise.\n\n" \
                        "- Your task is NOT to answer the question.\n" \
                        "- Instead, identify the incorrect scene premise in the question and state that it is wrong.\n" \
                        "- Then provide the correct scene premise explicitly.\n" \
                        "- Do not output anything else.\n\n" \
                        f"Question:**{question}**\nCorrect_premise:**{premise}**\n\n" \
                        "Examples:\n" \
                        "- Question: 'The farmer is harvesting the crops, what tools might he use?'\n" \
                        "- Correct premise: farmer is planting crops\n" \
                        "- Output: 'The farmer is not harvesting the crops. He is planting crops.'\n\n" \
                        "- Question: 'The audience is leaving the stadium, which direction would the traffic flow?'\n" \
                        "- Correct premise: audience is entering\n" \
                        "- Output: 'The audience is not leaving the stadium. They are entering.'\n\n" \
                        "- Question: 'The sky is bright and sunny, what outdoor activities might people enjoy?'\n" \
                        "- Correct premise: sky is dark and cloudy\n" \
                        "- Output: 'The sky is not bright and sunny. It is dark and cloudy.'\n"
                return answer_prompt
            
            case "Logical":
                answer_prompt = "You are a reasoning model. You will be given a question that contains a false logical sequence premise.\n\n" \
                            "- Your task is NOT to answer the question.\n" \
                            "- Instead, identify the incorrect logical sequence in the question and state that it is wrong.\n" \
                            "- Then provide the correct logical premise explicitly.\n" \
                            "- Do not output anything else.\n\n" \
                            f"Question:**{question}**\nCorrect_premise:**{premise}**\n\n" \
                            "Examples:\n" \
                            "- Question: 'The ice cream melted first and then fell, what would a child do after seeing it?'\n" \
                            "- Correct premise: ice cream fell → it melted\n" \
                            "- Output: 'The ice cream did not melt first and then fall. It fell first and then melted.'\n\n" \
                            "- Question: 'People used umbrellas, so the weather became hot, how would this affect outdoor activities?'\n" \
                            "- Correct premise: weather is hot → people use umbrellas\n" \
                            "- Output: 'People using umbrellas did not cause the weather to become hot. The weather is hot, so people use umbrellas.'\n"
                return answer_prompt
            
            case "Commonsense":
                answer_prompt = "You are a reasoning model. You will be given a question that contains a false commonsense premise.\n\n" \
                                "- Your task is NOT to answer the question.\n" \
                                "- Instead, identify the incorrect commonsense premise in the question and state that it is wrong.\n" \
                                "- Then provide the correct commonsense premise explicitly.\n" \
                                "- Do not output anything else.\n\n" \
                                f"Question:**{question}**\nCorrect_premise:**{premise}**\n\n" \
                                "Examples:\n" \
                                "- Question: 'The apple is flying up to the tree, what would happen if a child tried to catch it?'\n" \
                                "- Correct premise: apple falls to ground\n" \
                                "- Output: 'The apple does not fly up to the tree. It falls to the ground.'\n\n" \
                                "- Question: 'The boat is driving on the road, which vehicles might it meet?'\n" \
                                "- Correct premise: boat floats on water\n" \
                                "- Output: 'The boat is not driving on the road. It floats on water.'\n\n" \
                                "- Question: 'The sun is rising in the west, how would people adjust their morning routines?'\n" \
                                "- Correct premise: sun rises in east\n" \
                                "- Output: 'The sun is not rising in the west. It rises in the east.'\n"
                return answer_prompt





    