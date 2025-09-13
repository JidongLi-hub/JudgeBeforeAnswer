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

    def get_real_answer_prompt(self, question=None, premise=None):
        real_answer_prompt = "You are a reasoning model. You will be given a question that contains a correct premise.\n\n" \
                            "- Your task is to answer the question directly.\n" \
                            "- First, verify that the premise in the question is correct based on what you observe in the image.\n" \
                            "- Then provide a comprehensive answer to the question.\n" \
                            "- You must output in the format: <think>xxxxx</think><answer>xxxxx</answer>\n\n" \
                            "Format Requirements:\n" \
                            "- <think></think>: Your reasoning process (maximum 150 words)\n" \
                            "  1. First, briefly describe what you observe in the image\n" \
                            "  2. Then analyze the premise mentioned in the question\n" \
                            "  3. Confirm that the premise is correct and matches your observation\n" \
                            "  4. Consider how to answer the question based on the correct premise\n" \
                            "  5. Formulate the appropriate answer\n" \
                            "- <answer></answer>: Your direct response to the question (maximum 50 words)\n" \
                            "  - Provide a clear and comprehensive answer to the question asked\n\n" \
                            f"Question: **{question}**\n" \
                            f"Correct_premise: **{premise}**\n\n" \
                            "Examples:\n\n" \
                            "Example 1:\n" \
                            "Question: \"The person is holding the horse, how would people react to this scene?\"\n" \
                            "Correct_premise: person is holding horse\n\n" \
                            "<think>\n" \
                            "Looking at the image, I can see a person who appears to be holding or leading a horse, with their hand on the horse's halter or reins. The question states \"the person is holding the horse,\" and this premise is accurate based on my observation. The person is indeed in physical contact with and controlling the horse, which matches the stated premise perfectly. Now I need to consider how people would typically react to seeing someone holding a horse. This is a normal interaction between humans and horses, so reactions would likely be neutral to positive, perhaps showing interest or admiration.\n" \
                            "</think>\n\n" \
                            "<answer>\n" \
                            "People would likely react positively to this scene. They might find it interesting or charming to see someone interacting with a horse, especially in an urban or suburban setting where horses are less common. Some might stop to watch, take photos, or ask questions about the horse. Children would probably be particularly excited and curious. Overall, the reaction would be one of interest and appreciation for the animal.\n" \
                            "</answer>\n\n" \
                            "Example 2:\n" \
                            "Question: \"There are two apples on the table. Are they ripe?\"\n" \
                            "Correct_premise: two apples\n\n" \
                            "<think>\n" \
                            "In the image, I can observe a table with apples on it. Counting carefully, I can see two apples positioned on the table surface. The question mentions \"two apples on the table,\" and this count is accurate based on my visual observation. The premise is correct. Now I need to assess whether these apples appear ripe. I should examine their color, texture, and overall appearance to determine their ripeness. Ripe apples typically have vibrant colors, smooth skin, and appear firm and healthy.\n" \
                            "</think>\n\n" \
                            "<answer>\n" \
                            "Yes, the two apples on the table appear to be ripe. They display vibrant colors and have smooth, healthy-looking skin without visible blemishes or soft spots. Their appearance suggests they are at peak ripeness and would be good to eat.\n" \
                            "</answer>\n\n" \
                            "Example 3:\n" \
                            "Question: \"The door is open, what might someone do when entering the room?\"\n" \
                            "Correct_premise: open door\n\n" \
                            "<think>\n" \
                            "Looking at the image, I can see a door that is clearly in an open position, allowing access to the room beyond. The question states \"the door is open,\" and this observation is accurate based on what I can see. The premise is correct - the door is indeed open. Now I need to consider what someone would typically do when entering a room through an open door. Common actions would include walking through, looking around to assess the space, turning on lights if needed, or proceeding to their intended destination within the room.\n" \
                            "</think>\n\n" \
                            "<answer>\n" \
                            "When entering the room through the open door, someone would likely walk through the doorway, pause to look around and get oriented with the space, possibly turn on lights if the room is dim, and then proceed to wherever they need to go in the room or begin whatever activity they came to do.\n" \
                            "</answer>"
        return real_answer_prompt

    def get_answer_prompt(self, question=None, premise=None):
        match self.q_type:
            case "Entity Existence":
                answer_prompt = "You are a reasoning model. You will be given a question that contains a false premise.\n\n" \
                            "- Your task is NOT to answer the question.\n" \
                            "- Instead, you must identify the incorrect premise in the question and state that it is wrong.\n" \
                            "- You must output in the format: <think>xxxxx</think><answer>xxxxx</answer>\n\n" \
                            "Format Requirements:\n" \
                            "- <think></think>: Your reasoning process (maximum 150 words)\n" \
                            "  1. First, briefly describe what you observe in the image\n" \
                            "  2. Then analyze the premise in the question\n" \
                            "  3. Identify why the premise is incorrect\n" \
                            "  4. Conclude that you should correct the premise rather than answer the question\n" \
                            "- <answer></answer>: Your final response (maximum 50 words)\n" \
                            "  - Point out the incorrect premise in the question\n" \
                            "  - Provide the correct premise explicitly\n\n" \
                            f"Question: **{question}**\n" \
                            f"Correct_premise: **{premise}**\n\n" \
                            "Examples:\n\n" \
                            "Example 1:\n" \
                            "Question: \"What color is the dog on the floor?\"\n" \
                            "Correct_premise: cat\n\n" \
                            "<think>\n" \
                            "Looking at the image, I can see an animal on the floor. However, this animal appears to be a cat, not a dog. It has typical feline features such as pointed ears, whiskers, and a cat-like body structure. The question assumes there is a dog on the floor, but this premise is incorrect based on what I observe. The animal is clearly a cat. Instead of answering about the color, I need to correct this fundamental misidentification.\n" \
                            "</think>\n\n" \
                            "<answer>\n" \
                            "There is no dog on the floor. The animal on the floor is actually a cat.\n" \
                            "</answer>\n\n" \
                            "Example 2:\n" \
                            "Question: \"These people are holding up an umbrella. Is it raining?\"\n" \
                            "Correct_premise: tent\n\n" \
                            "<think>\n" \
                            "In the image, I observe people engaged in an outdoor activity. They appear to be setting up or handling what looks like camping equipment. The objects they are working with have the characteristics of a tent - with fabric material, support structures, and the typical setup process of camping gear. The question incorrectly identifies this as an umbrella. Umbrellas and tents are quite different in size, structure, and purpose. Rather than speculating about weather, I should correct this misidentification.\n" \
                            "</think>\n\n" \
                            "<answer>\n" \
                            "The people are not holding up an umbrella. They are setting up a tent.\n" \
                            "</answer>\n\n" \
                            "Example 3:\n" \
                            "Question: \"Why is the woman wearing a hat in the kitchen?\"\n" \
                            "Correct_premise: man\n\n" \
                            "<think>\n" \
                            "Examining the image, I can see a person in what appears to be a kitchen setting. However, looking at the person's physical characteristics, clothing, and overall appearance, this appears to be a man rather than a woman. The question assumes the person is a woman, but this gender identification seems incorrect based on the visual evidence. The question asks about why a woman is wearing a hat, but since the person appears to be male, I should correct this premise rather than explain the hat-wearing behavior.\n" \
                            "</think>\n\n" \
                            "<answer>\n" \
                            "The person in the kitchen is not a woman. It is a man wearing a hat.\n" \
                            "</answer>"
                return answer_prompt
            
            case "Visual Attributes":
                answer_prompt = "You are a reasoning model. You will be given a question that contains a false visual attribute premise.\n\n" \
                            "- Your task is NOT to answer the question.\n" \
                            "- Instead, you must identify the incorrect visual attribute in the question and state that it is wrong.\n" \
                            "- You must output in the format: <think>xxxxx</think><answer>xxxxx</answer>\n\n" \
                            "Format Requirements:\n" \
                            "- <think></think>: Your reasoning process (maximum 150 words)\n" \
                            "  1. First, briefly describe what you observe in the image\n" \
                            "  2. Then analyze the visual attribute mentioned in the question\n" \
                            "  3. Identify why the attribute is incorrect\n" \
                            "  4. Conclude that you should correct the attribute rather than answer the question\n" \
                            "- <answer></answer>: Your final response (maximum 50 words)\n" \
                            "  - Point out the incorrect visual attribute in the question\n" \
                            "  - Provide the correct visual attribute explicitly\n\n" \
                            f"Question: **{question}**\n" \
                            f"Correct_premise: **{premise}**\n\n" \
                            "Examples:\n\n" \
                            "Example 1:\n" \
                            "Question: \"What shape is the green apple on the table?\"\n" \
                            "Correct_premise: red apple\n\n" \
                            "<think>\n" \
                            "Looking at the image, I can see an apple on the table. Examining its color closely, the apple appears to be red rather than green. The question asks about a \"green apple,\" but this color description is incorrect based on what I observe. The apple has the typical red coloration with possible variations in shade, but it is definitely not green. Instead of answering about the shape, I need to correct this color misidentification first.\n" \
                            "</think>\n\n" \
                            "<answer>\n" \
                            "There is no green apple on the table. The apple is red.\n" \
                            "</answer>\n\n" \
                            "Example 2:\n" \
                            "Question: \"Is the round box heavy?\"\n" \
                            "Correct_premise: square box\n\n" \
                            "<think>\n" \
                            "In the image, I can see a box. Examining its shape, the box appears to have straight edges and corners, making it square or rectangular rather than round. The question refers to a \"round box,\" but this shape description is inaccurate. Round objects are circular, while this box has angular, geometric edges typical of square or rectangular boxes. Rather than speculating about its weight, I should correct this shape misattribution.\n" \
                            "</think>\n\n" \
                            "<answer>\n" \
                            "The box is not round. The box is square.\n" \
                            "</answer>\n\n" \
                            "Example 3:\n" \
                            "Question: \"Which metal chair is closer to the window?\"\n" \
                            "Correct_premise: wooden chair\n\n" \
                            "<think>\n" \
                            "Looking at the image, I can see a chair near a window. Examining the chair's material and texture, it appears to be made of wood rather than metal. The wood grain, color, and surface characteristics are clearly visible, indicating natural wooden construction. The question assumes the chair is made of metal, but this material identification is incorrect. Wood and metal have distinctly different visual properties, and this chair shows wooden characteristics. I should correct this material misidentification.\n" \
                            "</think>\n\n" \
                            "<answer>\n" \
                            "The chair is not made of metal. It is a wooden chair.\n" \
                            "</answer>"
                return answer_prompt
            
            case "Numeric Attributes":
                answer_prompt = "You are a reasoning model. You will be given a question that contains a false numeric attribute premise.\n\n" \
                            "- Your task is NOT to answer the question.\n" \
                            "- Instead, you must identify the incorrect numeric attribute in the question and state that it is wrong.\n" \
                            "- You must output in the format: <think>xxxxx</think><answer>xxxxx</answer>\n\n" \
                            "Format Requirements:\n" \
                            "- <think></think>: Your reasoning process (maximum 150 words)\n" \
                            "  1. First, briefly describe what you observe in the image\n" \
                            "  2. Then analyze the numeric attribute mentioned in the question\n" \
                            "  3. Count or identify the correct quantity\n" \
                            "  4. Conclude that you should correct the count rather than answer the question\n" \
                            "- <answer></answer>: Your final response (maximum 50 words)\n" \
                            "  - Point out the incorrect numeric attribute in the question\n" \
                            "  - Provide the correct numeric attribute explicitly\n\n" \
                            f"Question: **{question}**\n" \
                            f"Correct_premise: **{premise}**\n\n" \
                            "Examples:\n\n" \
                            "Example 1:\n" \
                            "Question: \"There are three apples on the table. Are they ripe?\"\n" \
                            "Correct_premise: two apples\n\n" \
                            "<think>\n" \
                            "Looking at the image, I can see a table with apples on it. Counting carefully, I observe two apples placed on the table surface. The question states \"there are three apples,\" but this count is incorrect based on my visual observation. I can clearly distinguish two separate apple objects, not three. The question asks about ripeness, but before addressing that, I need to correct the fundamental counting error in the premise.\n" \
                            "</think>\n\n" \
                            "<answer>\n" \
                            "There are not three apples on the table. There are two apples.\n" \
                            "</answer>\n\n" \
                            "Example 2:\n" \
                            "Question: \"What's in these five boxes?\"\n" \
                            "Correct_premise: three boxes\n\n" \
                            "<think>\n" \
                            "In the image, I can see several boxes. Counting them systematically, I observe three distinct boxes arranged in the scene. The question refers to \"five boxes,\" but this numerical count is inaccurate. I can clearly identify three separate box objects, each with distinct boundaries and positions. The question asks about the contents, but first I must address the incorrect quantity mentioned in the premise.\n" \
                            "</think>\n\n" \
                            "<answer>\n" \
                            "There are not five boxes. There are three boxes.\n" \
                            "</answer>\n\n" \
                            "Example 3:\n" \
                            "Question: \"Can students use these two chairs in the room?\"\n" \
                            "Correct_premise: several chairs\n\n" \
                            "<think>\n" \
                            "Examining the image, I can see a room with chairs. Counting the chairs visible in the scene, there appear to be more than two chairs present. I can identify several chairs arranged in the room, significantly more than the \"two chairs\" mentioned in the question. The exact count appears to be multiple chairs, which would be described as \"several.\" Rather than answering about student usage, I should correct this quantitative misstatement first.\n" \
                            "</think>\n\n" \
                            "<answer>\n" \
                            "There are not two chairs in the room. There are several chairs.\n" \
                            "</answer>"
                return answer_prompt

            case "State Attributes":
                answer_prompt = "You are a reasoning model. You will be given a question that contains a false state attribute premise.\n\n" \
                            "- Your task is NOT to answer the question.\n" \
                            "- Instead, you must identify the incorrect state attribute in the question and state that it is wrong.\n" \
                            "- You must output in the format: <think>xxxxx</think><answer>xxxxx</answer>\n\n" \
                            "Format Requirements:\n" \
                            "- <think></think>: Your reasoning process (maximum 150 words)\n" \
                            "  1. First, briefly describe what you observe in the image\n" \
                            "  2. Then analyze the state attribute mentioned in the question\n" \
                            "  3. Identify the actual state and why the premise is incorrect\n" \
                            "  4. Conclude that you should correct the state rather than answer the question\n" \
                            "- <answer></answer>: Your final response (maximum 50 words)\n" \
                            "  - Point out the incorrect state attribute in the question\n" \
                            "  - Provide the correct state attribute explicitly\n\n" \
                            f"Question: **{question}**\n" \
                            f"Correct_premise: **{premise}**\n\n" \
                            "Examples:\n\n" \
                            "Example 1:\n" \
                            "Question: \"What color is the closed door?\"\n" \
                            "Correct_premise: open door\n\n" \
                            "<think>\n" \
                            "Looking at the image, I can see a door in the scene. Examining its current state, the door appears to be in an open position rather than closed. I can observe the door frame and the opening that indicates the door is ajar or fully opened. The question refers to a \"closed door,\" but this state description is incorrect based on the visual evidence. The door is clearly not in a closed position. Rather than answering about the color, I need to correct this state misidentification.\n" \
                            "</think>\n\n" \
                            "<answer>\n" \
                            "The door is not closed. It is an open door.\n" \
                            "</answer>\n\n" \
                            "Example 2:\n" \
                            "Question: \"Which shelf is the intact vase on?\"\n" \
                            "Correct_premise: broken vase\n\n" \
                            "<think>\n" \
                            "In the image, I can see a vase positioned on what appears to be a shelf or surface. Examining the vase carefully, I notice visible damage such as cracks, chips, or broken pieces, indicating that the vase is not intact but rather damaged or broken. The question assumes the vase is \"intact,\" but this condition assessment is incorrect. An intact vase would show no signs of damage, while this vase clearly shows evidence of being broken or damaged. I should correct this condition misstatement.\n" \
                            "</think>\n\n" \
                            "<answer>\n" \
                            "The vase is not intact. It is a broken vase.\n" \
                            "</answer>\n\n" \
                            "Example 3:\n" \
                            "Question: \"How bright is the unlit lamp?\"\n" \
                            "Correct_premise: lit lamp\n\n" \
                            "<think>\n" \
                            "Looking at the image, I can see a lamp in the scene. Examining its current state, the lamp appears to be illuminated and giving off light, indicating that it is turned on or lit. I can observe the glow or brightness emanating from the lamp, which clearly shows it is in an active, illuminated state. The question refers to an \"unlit lamp,\" but this operational state is incorrect. An unlit lamp would show no illumination, while this lamp is clearly producing light. I should correct this state error.\n" \
                            "</think>\n\n" \
                            "<answer>\n" \
                            "The lamp is not unlit. It is a lit lamp.\n" \
                            "</answer>"
                return answer_prompt

            case "OCR Content":
                answer_prompt = "You are a reasoning model. You will be given a question that contains a false OCR text premise.\n\n" \
                            "- Your task is NOT to answer the question.\n" \
                            "- Instead, you must identify the incorrect text content in the question and state that it is wrong.\n" \
                            "- You must output in the format: <think>xxxxx</think><answer>xxxxx</answer>\n\n" \
                            "Format Requirements:\n" \
                            "- <think></think>: Your reasoning process (maximum 150 words)\n" \
                            "  1. First, briefly describe what you observe in the image\n" \
                            "  2. Then analyze the text content mentioned in the question\n" \
                            "  3. Read the actual text shown and identify the discrepancy\n" \
                            "  4. Conclude that you should correct the text rather than answer the question\n" \
                            "- <answer></answer>: Your final response (maximum 50 words)\n" \
                            "  - Point out the incorrect text content in the question\n" \
                            "  - Provide the correct text content explicitly\n\n" \
                            f"Question: **{question}**\n" \
                            f"Correct_premise: **{premise}**\n\n" \
                            "Examples:\n\n" \
                            "Example 1:\n" \
                            "Question: \"What color is the sign with the word GO?\"\n" \
                            "Correct_premise: STOP\n\n" \
                            "<think>\n" \
                            "Looking at the image, I can see a sign with text on it. Reading the text carefully, the sign displays the word \"STOP\" rather than \"GO.\" The question refers to a sign with the word \"GO,\" but this text identification is incorrect based on what I can actually read from the sign. STOP and GO are completely different words with opposite meanings. Rather than answering about the color, I need to correct this fundamental text misreading first.\n" \
                            "</think>\n\n" \
                            "<answer>\n" \
                            "The sign does not show GO. It shows STOP.\n" \
                            "</answer>\n\n" \
                            "Example 2:\n" \
                            "Question: \"How many stars are shown near the sign with the word CAFE?\"\n" \
                            "Correct_premise: HOTEL\n\n" \
                            "<think>\n" \
                            "In the image, I can see a sign with text on it. Reading the text carefully, the sign displays the word \"HOTEL\" not \"CAFE.\" The question assumes the sign says \"CAFE,\" but this text reading is incorrect. HOTEL and CAFE are different types of business establishments, and the actual text shown is clearly \"HOTEL.\" Instead of counting stars, I should first correct this text misidentification that forms the basis of the question.\n" \
                            "</think>\n\n" \
                            "<answer>\n" \
                            "The sign does not show CAFE. It shows HOTEL.\n" \
                            "</answer>\n\n" \
                            "Example 3:\n" \
                            "Question: \"Is the board with the word TOYS hanging outside the shop?\"\n" \
                            "Correct_premise: BOOKS\n\n" \
                            "<think>\n" \
                            "Looking at the image, I can see a board or sign with text displayed on it. Reading the text carefully, the board shows the word \"BOOKS\" rather than \"TOYS.\" The question refers to a board with the word \"TOYS,\" but this text identification is incorrect based on my reading of the actual text shown. BOOKS and TOYS represent different types of merchandise or shop categories. Rather than answering about its hanging position, I should correct this text reading error.\n" \
                            "</think>\n\n" \
                            "<answer>\n" \
                            "The board does not show TOYS. It shows BOOKS.\n" \
                            "</answer>"
                return answer_prompt

            case "Symbol Meaning":
                answer_prompt = "You are a reasoning model. You will be given a question that contains a false symbol premise.\n\n" \
                            "- Your task is NOT to answer the question.\n" \
                            "- Instead, you must identify the incorrect symbol in the question and state that it is wrong.\n" \
                            "- You must output in the format: <think>xxxxx</think><answer>xxxxx</answer>\n\n" \
                            "Format Requirements:\n" \
                            "- <think></think>: Your reasoning process (maximum 150 words)\n" \
                            "  1. First, briefly describe what you observe in the image\n" \
                            "  2. Then analyze the symbol or sign mentioned in the question\n" \
                            "  3. Identify the actual symbol and why the premise is incorrect\n" \
                            "  4. Conclude that you should correct the symbol rather than answer the question\n" \
                            "- <answer></answer>: Your final response (maximum 50 words)\n" \
                            "  - Point out the incorrect symbol in the question\n" \
                            "  - Provide the correct symbol explicitly\n\n" \
                            f"Question: **{question}**\n" \
                            f"Correct_premise: **{premise}**\n\n" \
                            "Examples:\n\n" \
                            "Example 1:\n" \
                            "Question: \"What does the No Parking sign mean at the corner?\"\n" \
                            "Correct_premise: STOP sign\n\n" \
                            "<think>\n" \
                            "Looking at the image, I can see a traffic sign at the corner location. Examining the sign carefully, it displays the characteristic octagonal red shape with white text that reads \"STOP,\" which is a standard STOP sign. The question refers to a \"No Parking sign,\" but this symbol identification is incorrect. A No Parking sign would have different visual characteristics, typically showing a \"P\" with a diagonal line through it. The actual sign is clearly a STOP sign. Rather than explaining its meaning, I should correct this symbol misidentification.\n" \
                            "</think>\n\n" \
                            "<answer>\n" \
                            "There is no No Parking sign at the corner. It is a STOP sign.\n" \
                            "</answer>\n\n" \
                            "Example 2:\n" \
                            "Question: \"Is the Left Turn sign visible on the road?\"\n" \
                            "Correct_premise: Right Turn sign\n\n" \
                            "<think>\n" \
                            "In the image, I can see a traffic sign on or near the road. Examining the sign's symbol and arrow direction, it shows an arrow pointing to the right, indicating a right turn direction. The question asks about a \"Left Turn sign,\" but this directional identification is incorrect. Left turn signs would show arrows pointing to the left, while this sign clearly displays a right-pointing arrow. The visual symbol definitively indicates a right turn, not a left turn. I should correct this directional symbol error.\n" \
                            "</think>\n\n" \
                            "<answer>\n" \
                            "There is no Left Turn sign on the road. It is a Right Turn sign.\n" \
                            "</answer>\n\n" \
                            "Example 3:\n" \
                            "Question: \"Which direction does the Pedestrian Crossing sign point to?\"\n" \
                            "Correct_premise: Warning sign\n\n" \
                            "<think>\n" \
                            "Looking at the image, I can see a sign near what appears to be a construction area. Examining the sign's symbol and design, it displays warning indicators such as caution symbols or construction-related imagery, characteristic of a general warning sign. The question refers to a \"Pedestrian Crossing sign,\" but this symbol identification is incorrect. Pedestrian crossing signs typically show figures of people walking, while this appears to be a warning sign related to construction or hazards. I should correct this symbol category error.\n" \
                            "</think>\n\n" \
                            "<answer>\n" \
                            "There is no Pedestrian Crossing sign. It is a Warning sign.\n" \
                            "</answer>"
                return answer_prompt
            
            case "Spatial Relation":
                answer_prompt = "You are a reasoning model. You will be given a question that contains a false spatial relation premise.\n\n" \
                            "- Your task is NOT to answer the question.\n" \
                            "- Instead, you must identify the incorrect spatial relation in the question and state that it is wrong.\n" \
                            "- You must output in the format: <think>xxxxx</think><answer>xxxxx</answer>\n\n" \
                            "Format Requirements:\n" \
                            "- <think></think>: Your reasoning process (maximum 150 words)\n" \
                            "  1. First, briefly describe what you observe in the image\n" \
                            "  2. Then analyze the spatial relationship mentioned in the question\n" \
                            "  3. Identify the actual spatial relationship and why the premise is incorrect\n" \
                            "  4. Conclude that you should correct the spatial relation rather than answer the question\n" \
                            "- <answer></answer>: Your final response (maximum 50 words)\n" \
                            "  - Point out the incorrect spatial relation in the question\n" \
                            "  - Provide the correct spatial relation explicitly\n\n" \
                            f"Question: **{question}**\n" \
                            f"Correct_premise: **{premise}**\n\n" \
                            "Examples:\n\n" \
                            "Example 1:\n" \
                            "Question: \"The apple is under the table, what might the child do to find it?\"\n" \
                            "Correct_premise: apple on table\n\n" \
                            "<think>\n" \
                            "Looking at the image, I can see a table and an apple. Examining their spatial relationship, the apple appears to be positioned on top of the table surface, not underneath it. The question states \"the apple is under the table,\" but this spatial description is incorrect based on my observation. \"Under\" would mean below the table, while \"on\" indicates the apple is resting on the table's surface. The actual positioning shows the apple is clearly on the table. Rather than suggesting search strategies, I should correct this spatial misidentification.\n" \
                            "</think>\n\n" \
                            "<answer>\n" \
                            "The apple is not under the table. It is on the table.\n" \
                            "</answer>\n\n" \
                            "Example 2:\n" \
                            "Question: \"The river is on the right of the tree, which path would animals take to drink water?\"\n" \
                            "Correct_premise: river left of tree\n\n" \
                            "<think>\n" \
                            "In the image, I can observe a tree and a river. Analyzing their relative positions, the river appears to be located to the left side of the tree when viewed from this perspective. The question claims \"the river is on the right of the tree,\" but this directional relationship is incorrect. Left and right are opposite spatial orientations, and the visual evidence clearly shows the river positioned on the left side of the tree. I should correct this spatial orientation error rather than discuss animal paths.\n" \
                            "</think>\n\n" \
                            "<answer>\n" \
                            "The river is not on the right of the tree. It is on the left of the tree.\n" \
                            "</answer>\n\n" \
                            "Example 3:\n" \
                            "Question: \"The cake is behind the box, what would you see first when entering the room?\"\n" \
                            "Correct_premise: cake next to box\n\n" \
                            "<think>\n" \
                            "Looking at the image, I can see a cake and a box. Examining their spatial arrangement, the cake appears to be positioned alongside or adjacent to the box rather than behind it. The question states \"the cake is behind the box,\" but this positional relationship is incorrect. \"Behind\" would mean the cake is obscured or positioned at the back of the box, while \"next to\" indicates they are side by side. The actual positioning shows they are adjacent to each other. I should correct this spatial relationship error.\n" \
                            "</think>\n\n" \
                            "<answer>\n" \
                            "The cake is not behind the box. It is next to the box.\n" \
                            "</answer>"
                return answer_prompt

            case "Interaction Relation":
                answer_prompt ="You are a reasoning model. You will be given a question that contains a false interaction relation premise.\n\n" \
                            "- Your task is NOT to answer the question.\n" \
                            "- Instead, you must identify the incorrect interaction relation in the question and state that it is wrong.\n" \
                            "- You must output in the format: <think>...</think><answer>...</answer>\n\n" \
                            "Format Requirements:\n" \
                            "- <think></think>: Your reasoning process (maximum 150 words)\n" \
                            "  1. First, briefly describe what you observe in the image\n" \
                            "  2. Then analyze the premise in the question\n" \
                            "  3. Identify why the premise is incorrect\n" \
                            "  4. Conclude that you should correct the premise rather than answer the question\n" \
                            "- <answer></answer>: Your final response (maximum 50 words)\n" \
                            "  - Point out the incorrect premise in the question\n" \
                            "  - Provide the correct interaction relation\n\n" \
                            f"Question: **{question}**\n" \
                            f"Correct_premise: **{premise}**\n\n" \
                            "Examples:\n\n" \
                            "Example 1:\n" \
                            "Question: \"The horse is holding the person, how would people react?\"\n" \
                            "Correct_premise: person is holding horse\n\n" \
                            "<think>\n" \
                            "I can see an image showing a person and a horse. The person appears to be holding or leading the horse, with the person's hand on the horse's halter or reins. The question states \"the horse is holding the person,\" but this is physically impossible and contradicts what I observe. Horses do not have hands to hold people. The actual relationship shows the person holding/controlling the horse. Rather than answering how people would react, I need to correct this false premise.\n" \
                            "</think>\n\n" \
                            "<answer>\n" \
                            "The premise is incorrect. The horse is not holding the person. The correct relationship is that the person is holding the horse.\n" \
                            "</answer>\n\n" \
                            "Example 2:\n" \
                            "Question: \"The pigeon is chasing the girl, what might happen next?\"\n" \
                            "Correct_premise: girl is chasing pigeon\n\n" \
                            "<think>\n" \
                            "Looking at the image, I can see a girl and a pigeon. The girl appears to be moving toward or pursuing the pigeon, while the pigeon seems to be moving away from her. The question claims \"the pigeon is chasing the girl,\" but this reverses the actual direction of pursuit. Typically, pigeons flee from humans rather than chase them. The visual evidence shows the girl is the one doing the chasing. I should correct this misstatement rather than speculate about what happens next.\n" \
                            "</think>\n\n" \
                            "<answer>\n" \
                            "The premise is wrong. The pigeon is not chasing the girl. The girl is chasing the pigeon.\n" \
                            "</answer>\n\n" \
                            "Example 3:\n" \
                            "Question: \"The cat is being petted by the mouse, why does this seem unusual?\"\n" \
                            "Correct_premise: cat is petting mouse\n\n" \
                            "<think>\n" \
                            "In the image, I observe a cat and a mouse. The cat appears to be gently touching or pawing at the mouse, which would be the cat petting/touching the mouse. The question states \"the cat is being petted by the mouse,\" which reverses the roles. This is biologically implausible as mice are much smaller than cats and typically avoid cats rather than pet them. The actual interaction shows the cat as the active agent touching the mouse. I need to correct this false premise.\n" \
                            "</think>\n\n" \
                            "<answer>\n" \
                            "The premise is incorrect. The cat is not being petted by the mouse. The cat is petting the mouse.\n" \
                            "</answer>"                
                return answer_prompt

            case "Possessive Relation":
                answer_prompt = "You are a reasoning model. You will be given a question that contains a false possessive relation premise.\n\n" \
                        "- Your task is NOT to answer the question.\n" \
                        "- Instead, you must identify the incorrect possessive relation in the question and state that it is wrong.\n" \
                        "- You must output in the format: <think>xxxxx</think><answer>xxxxx</answer>\n\n" \
                        "Format Requirements:\n" \
                        "- <think></think>: Your reasoning process (maximum 150 words)\n" \
                        "  1. First, briefly describe what you observe in the image\n" \
                        "  2. Then analyze the possessive/causal relationship mentioned in the question\n" \
                        "  3. Identify the actual relationship and why the premise is incorrect\n" \
                        "  4. Conclude that you should correct the relationship rather than answer the question\n" \
                        "- <answer></answer>: Your final response (maximum 50 words)\n" \
                        "  - Point out the incorrect possessive relation in the question\n" \
                        "  - Provide the correct relation explicitly\n\n" \
                        f"Question: **{question}**\n" \
                        f"Correct_premise: **{premise}**\n\n" \
                        "Examples:\n\n" \
                        "Example 1:\n" \
                        "Question: \"If the cart is pulling the horse, how far could they travel together?\"\n" \
                        "Correct_premise: horse pulls cart\n\n" \
                        "<think>\n" \
                        "Looking at the image, I can see a horse and a cart. Examining their relationship, the horse appears to be positioned in front of the cart, connected by harnesses or reins, which indicates the horse is the one providing the pulling force. The question states \"the cart is pulling the horse,\" but this causal relationship is reversed. In reality, horses are the animals that provide power and pull carts, not the other way around. This is a fundamental reversal of the logical cause-and-effect relationship. I should correct this before discussing travel distance.\n" \
                        "</think>\n\n" \
                        "<answer>\n" \
                        "The cart is not pulling the horse. The horse pulls the cart.\n" \
                        "</answer>\n\n" \
                        "Example 2:\n" \
                        "Question: \"If the castle builds the bricks, how strong will the structure be?\"\n" \
                        "Correct_premise: bricks build castle\n\n" \
                        "<think>\n" \
                        "In the image, I can observe a castle structure and bricks. Analyzing the construction relationship, bricks are the building materials used to construct the castle, not the other way around. The question states \"the castle builds the bricks,\" but this construction relationship is illogical and reversed. Castles are built from bricks and other materials; they don't create their own building materials. This represents a fundamental misunderstanding of the cause-and-effect relationship in construction. I should correct this logical error rather than assess structural strength.\n" \
                        "</think>\n\n" \
                        "<answer>\n" \
                        "The castle does not build the bricks. The bricks build the castle.\n" \
                        "</answer>\n\n" \
                        "Example 3:\n" \
                        "Question: \"If the book reads the student, what knowledge would be gained?\"\n" \
                        "Correct_premise: student reads book\n\n" \
                        "<think>\n" \
                        "Looking at the image, I can see a student and a book. Examining their relationship, the student appears to be the active agent holding or looking at the book, which indicates the student is reading the book. The question states \"the book reads the student,\" but this relationship is nonsensical and reversed. Books are inanimate objects that cannot read; students are the ones who read books to gain knowledge. This reverses the fundamental subject-object relationship in the learning process. I should correct this logical impossibility.\n" \
                        "</think>\n\n" \
                        "<answer>\n" \
                        "The book does not read the student. The student reads the book.\n" \
                        "</answer>"
                return answer_prompt

            case "Emotion":
                answer_prompt = "You are a reasoning model. You will be given a question that contains a false emotional state premise.\n\n" \
                            "- Your task is NOT to answer the question.\n" \
                            "- Instead, you must identify the incorrect emotional state in the question and state that it is wrong.\n" \
                            "- You must output in the format: <think>xxxxx</think><answer>xxxxx</answer>\n\n" \
                            "Format Requirements:\n" \
                            "- <think></think>: Your reasoning process (maximum 150 words)\n" \
                            "  1. First, briefly describe what you observe in the image\n" \
                            "  2. Then analyze the emotional state mentioned in the question\n" \
                            "  3. Identify the actual emotional state and why the premise is incorrect\n" \
                            "  4. Conclude that you should correct the emotional state rather than answer the question\n" \
                            "- <answer></answer>: Your final response (maximum 50 words)\n" \
                            "  - Point out the incorrect emotional state in the question\n" \
                            "  - Provide the correct emotional state explicitly\n\n" \
                            f"Question: **{question}**\n" \
                            f"Correct_premise: **{premise}**\n\n" \
                            "Examples:\n\n" \
                            "Example 1:\n" \
                            "Question: \"The man is joyful, what might he be celebrating?\"\n" \
                            "Correct_premise: man is sad\n\n" \
                            "<think>\n" \
                            "Looking at the image, I can see a man whose facial expression and body language suggest sadness rather than joy. His facial features appear downturned, his posture seems dejected, and his overall demeanor indicates a melancholic or sorrowful state. The question states \"the man is joyful,\" but this emotional assessment is incorrect based on the visual cues I observe. Joyful expressions typically involve smiling, upright posture, and bright eyes, which are not present here. Rather than speculating about celebrations, I should correct this emotional state misidentification.\n" \
                            "</think>\n\n" \
                            "<answer>\n" \
                            "The man is not joyful. He is sad.\n" \
                            "</answer>\n\n" \
                            "Example 2:\n" \
                            "Question: \"The woman is angry, how would her friends try to calm her down?\"\n" \
                            "Correct_premise: woman is joyful\n\n" \
                            "<think>\n" \
                            "In the image, I can observe a woman whose facial expression and body language indicate happiness and joy. She appears to be smiling, her eyes seem bright and positive, and her overall demeanor suggests a cheerful, joyful state. The question claims \"the woman is angry,\" but this emotional reading is incorrect. Angry expressions typically involve frowning, tense facial muscles, and aggressive body language, none of which are evident here. The woman clearly displays positive, joyful emotions. I should correct this emotional state error rather than suggest calming strategies.\n" \
                            "</think>\n\n" \
                            "<answer>\n" \
                            "The woman is not angry. She is joyful.\n" \
                            "</answer>\n\n" \
                            "Example 3:\n" \
                            "Question: \"The dog is calm, which game would it play with the children in this weather?\"\n" \
                            "Correct_premise: dog is fearful\n\n" \
                            "<think>\n" \
                            "Looking at the image, I can see a dog whose body language and behavior suggest fear rather than calmness. The dog appears to have a tense posture, possibly cowering or showing signs of anxiety such as lowered head, tucked tail, or trembling. These are indicators of fearfulness, not calmness. The question states \"the dog is calm,\" but this behavioral assessment is incorrect. Calm dogs typically display relaxed postures, neutral expressions, and confident stances, which are not present here. I should correct this emotional state misreading.\n" \
                            "</think>\n\n" \
                            "<answer>\n" \
                            "The dog is not calm. It is fearful.\n" \
                            "</answer>"
                return answer_prompt
            
            case "Scene":
                answer_prompt = "You are a reasoning model. You will be given a question that contains a false scene premise.\n\n" \
                            "- Your task is NOT to answer the question.\n" \
                            "- Instead, you must identify the incorrect scene premise in the question and state that it is wrong.\n" \
                            "- You must output in the format: <think>xxxxx</think><answer>xxxxx</answer>\n\n" \
                            "Format Requirements:\n" \
                            "- <think></think>: Your reasoning process (maximum 150 words)\n" \
                            "  1. First, briefly describe what you observe in the image\n" \
                            "  2. Then analyze the scene or activity mentioned in the question\n" \
                            "  3. Identify the actual scene/activity and why the premise is incorrect\n" \
                            "  4. Conclude that you should correct the scene rather than answer the question\n" \
                            "- <answer></answer>: Your final response (maximum 50 words)\n" \
                            "  - Point out the incorrect scene premise in the question\n" \
                            "  - Provide the correct scene premise explicitly\n\n" \
                            f"Question: **{question}**\n" \
                            f"Correct_premise: **{premise}**\n\n" \
                            "Examples:\n\n" \
                            "Example 1:\n" \
                            "Question: \"The farmer is harvesting the crops, what tools might he use?\"\n" \
                            "Correct_premise: farmer is planting crops\n\n" \
                            "<think>\n" \
                            "Looking at the image, I can see a farmer working in a field. Examining the activity more closely, the farmer appears to be engaged in planting or sowing activities, such as placing seeds in the soil or preparing the ground for planting. The question states \"the farmer is harvesting the crops,\" but this activity description is incorrect. Harvesting involves gathering mature crops, while planting involves putting seeds into the ground. These are opposite activities in the agricultural cycle. Rather than suggesting harvesting tools, I should correct this activity misidentification.\n" \
                            "</think>\n\n" \
                            "<answer>\n" \
                            "The farmer is not harvesting the crops. He is planting crops.\n" \
                            "</answer>\n\n" \
                            "Example 2:\n" \
                            "Question: \"The audience is leaving the stadium, which direction would the traffic flow?\"\n" \
                            "Correct_premise: audience is entering\n\n" \
                            "<think>\n" \
                            "In the image, I can observe people at a stadium setting. Analyzing their movement and body language, the crowd appears to be moving toward the stadium entrance, with people walking in the direction of the gates or entrances. The question claims \"the audience is leaving the stadium,\" but this movement description is incorrect. Leaving would involve people moving away from the stadium, while entering involves moving toward and into the venue. The visual evidence shows people approaching or entering the stadium. I should correct this directional activity error.\n" \
                            "</think>\n\n" \
                            "<answer>\n" \
                            "The audience is not leaving the stadium. They are entering.\n" \
                            "</answer>\n\n" \
                            "Example 3:\n" \
                            "Question: \"The sky is bright and sunny, what outdoor activities might people enjoy?\"\n" \
                            "Correct_premise: sky is dark and cloudy\n\n" \
                            "<think>\n" \
                            "Looking at the image, I can observe the sky and weather conditions. The sky appears to be overcast with dark, heavy clouds, and the overall lighting suggests gloomy or stormy weather conditions. The question describes the sky as \"bright and sunny,\" but this weather assessment is completely incorrect. Bright and sunny skies would show clear blue skies with visible sunshine, while this image shows the opposite - dark, cloudy conditions that suggest poor weather. I should correct this weather condition misidentification rather than suggest outdoor activities.\n" \
                            "</think>\n\n" \
                            "<answer>\n" \
                            "The sky is not bright and sunny. It is dark and cloudy.\n" \
                            "</answer>"
                return answer_prompt
            
            case "Logical":
                answer_prompt = "You are a reasoning model. You will be given a question that contains a false logical sequence premise.\n\n" \
                            "- Your task is NOT to answer the question.\n" \
                            "- Instead, you must identify the incorrect logical sequence in the question and state that it is wrong.\n" \
                            "- You must output in the format: <think>xxxxx</think><answer>xxxxx</answer>\n\n" \
                            "Format Requirements:\n" \
                            "- <think></think>: Your reasoning process (maximum 150 words)\n" \
                            "  1. First, briefly describe what you observe in the image\n" \
                            "  2. Then analyze the logical sequence or cause-effect relationship mentioned in the question\n" \
                            "  3. Identify the correct logical sequence and why the premise is incorrect\n" \
                            "  4. Conclude that you should correct the logical sequence rather than answer the question\n" \
                            "- <answer></answer>: Your final response (maximum 50 words)\n" \
                            "  - Point out the incorrect logical sequence in the question\n" \
                            "  - Provide the correct logical sequence explicitly\n\n" \
                            f"Question: **{question}**\n" \
                            f"Correct_premise: **{premise}**\n\n" \
                            "Examples:\n\n" \
                            "Example 1:\n" \
                            "Question: \"The ice cream melted first and then fell, what would a child do after seeing it?\"\n" \
                            "Correct_premise: ice cream fell → it melted\n\n" \
                            "<think>\n" \
                            "Looking at the image, I can see ice cream that appears to have fallen and is now melting. The question suggests the sequence was \"melted first and then fell,\" but this logical order is incorrect. Ice cream typically maintains its solid form when held properly. The logical sequence would be that the ice cream fell first (due to dropping, slipping, or losing grip), and then as a result of being on the ground or a warm surface, it began to melt. Melting is typically a consequence of the fall, not the cause. I should correct this cause-and-effect reversal.\n" \
                            "</think>\n\n" \
                            "<answer>\n" \
                            "The ice cream did not melt first and then fall. It fell first and then melted.\n" \
                            "</answer>\n\n" \
                            "Example 2:\n" \
                            "Question: \"People used umbrellas, so the weather became hot, how would this affect outdoor activities?\"\n" \
                            "Correct_premise: weather is hot → people use umbrellas\n\n" \
                            "<think>\n" \
                            "In the image, I can see people using umbrellas in what appears to be hot, sunny weather. The question suggests that \"people used umbrellas, so the weather became hot,\" but this causal relationship is reversed and illogical. Umbrellas don't cause weather changes; rather, weather conditions cause people to use umbrellas. In hot, sunny weather, people use umbrellas for shade and protection from the sun. The correct sequence is that hot weather leads to umbrella usage, not the other way around. I should correct this cause-and-effect error.\n" \
                            "</think>\n\n" \
                            "<answer>\n" \
                            "People using umbrellas did not cause the weather to become hot. The weather is hot, so people use umbrellas.\n" \
                            "</answer>\n\n" \
                            "Example 3:\n" \
                            "Question: \"The flowers bloomed because they were watered, then the rain started, what season might this be?\"\n" \
                            "Correct_premise: rain started → flowers were watered → flowers bloomed\n\n" \
                            "<think>\n" \
                            "Looking at the image, I can see blooming flowers in what appears to be a natural outdoor setting with signs of recent rainfall. The question suggests the sequence was \"flowers bloomed because they were watered, then rain started,\" but this temporal and causal order is incorrect. The logical sequence would be that rain started first, providing natural watering to the flowers, which then caused them to bloom. Rain is the natural cause that leads to watering, which then results in blooming. I should correct this sequence error.\n" \
                            "</think>\n\n" \
                            "<answer>\n" \
                            "The sequence is incorrect. The rain started first, which watered the flowers, then the flowers bloomed.\n" \
                            "</answer>"
                return answer_prompt
            
            case "Commonsense":
                answer_prompt = "You are a reasoning model. You will be given a question that contains a false commonsense premise.\n\n" \
                            "- Your task is NOT to answer the question.\n" \
                            "- Instead, you must identify the incorrect commonsense premise in the question and state that it is wrong.\n" \
                            "- You must output in the format: <think>xxxxx</think><answer>xxxxx</answer>\n\n" \
                            "Format Requirements:\n" \
                            "- <think></think>: Your reasoning process (maximum 150 words)\n" \
                            "  1. First, briefly describe what you observe in the image\n" \
                            "  2. Then analyze the commonsense premise mentioned in the question\n" \
                            "  3. Identify why the premise violates basic commonsense or natural laws\n" \
                            "  4. Conclude that you should correct the commonsense error rather than answer the question\n" \
                            "- <answer></answer>: Your final response (maximum 50 words)\n" \
                            "  - Point out the incorrect commonsense premise in the question\n" \
                            "  - Provide the correct commonsense premise explicitly\n\n" \
                            f"Question: **{question}**\n" \
                            f"Correct_premise: **{premise}**\n\n" \
                            "Examples:\n\n" \
                            "Example 1:\n" \
                            "Question: \"The apple is flying up to the tree, what would happen if a child tried to catch it?\"\n" \
                            "Correct_premise: apple falls to ground\n\n" \
                            "<think>\n" \
                            "Looking at the image, I can see an apple that appears to be falling or has fallen from a tree. The question states \"the apple is flying up to the tree,\" but this violates basic physics and commonsense knowledge. Gravity causes objects like apples to fall downward, not fly upward. Apples don't have any mechanism for flight and are subject to gravitational force. When apples detach from trees, they naturally fall to the ground due to gravity. This premise contradicts fundamental natural laws. I should correct this physics impossibility.\n" \
                            "</think>\n\n" \
                            "<answer>\n" \
                            "The apple does not fly up to the tree. It falls to the ground.\n" \
                            "</answer>\n\n" \
                            "Example 2:\n" \
                            "Question: \"The boat is driving on the road, which vehicles might it meet?\"\n" \
                            "Correct_premise: boat floats on water\n\n" \
                            "<think>\n" \
                            "In the image, I can see a boat in what appears to be a water environment. The question claims \"the boat is driving on the road,\" but this violates basic commonsense about transportation. Boats are designed to float on water using buoyancy principles and cannot operate on roads. Roads are for land vehicles with wheels, while boats require water for their hull design to function properly. Boats would be damaged and unable to move effectively on solid road surfaces. This premise contradicts basic transportation knowledge.\n" \
                            "</think>\n\n" \
                            "<answer>\n" \
                            "The boat is not driving on the road. It floats on water.\n" \
                            "</answer>\n\n" \
                            "Example 3:\n" \
                            "Question: \"The sun is rising in the west, how would people adjust their morning routines?\"\n" \
                            "Correct_premise: sun rises in east\n\n" \
                            "<think>\n" \
                            "Looking at the image showing what appears to be a sunrise or sunset scene, the question states \"the sun is rising in the west,\" but this contradicts fundamental astronomical knowledge. Due to Earth's rotation from west to east, the sun always appears to rise in the east and set in the west from any location on Earth. This is a universal, consistent phenomenon that has been observed throughout human history. The premise violates basic geographical and astronomical commonsense that is fundamental to human understanding of daily cycles.\n" \
                            "</think>\n\n" \
                            "<answer>\n" \
                            "The sun is not rising in the west. It rises in the east.\n" \
                            "</answer>"
                return answer_prompt
    
    
    def get_answer_prompt_old(self, question=None, premise=None):
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





    