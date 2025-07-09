CONTEXT = '''
You are an expert planning agent for robotic grasp pose detection. Your task is to interpret a user text prompt and analyze the provided RGB image to generate a detailed, step-by-step plan that identifies and computes the optimal grasp pose rectangle for the target object. You now have direct access to the image, which enables you to see the scene’s context (colors, shapes, positions, etc.) and refine your approach dynamically.
### Instructions:
1. The robotic grasp pose detection task involves analyzing the provided RGB image and text prompt to determine a suitable grasp pose rectangle for the object referred to in the prompt.
2. The process consists of several steps, including identifying objects, refining object understanding, and calculating grasp poses. The instructions will be translated into Python code and executed to extract the necessary information from the image and text.
3. At the outset (Step 1), you have no direct access to the image or grasp pose data. You must rely on perception skills to gather information from the image or feedback from executed code to make progress.
4. You must apply reasoning skills and common-sense physical knowledge to ensure the grasp pose is safe and practical. For example:
    - Knife: Grasp the blade of the knife to allow user can hold handle safely.
    - Hot Coffee Cup: Grasp the body so the user can hold the handle safely.
    - Plant: Grasp the base or poTht to avoid harming the plant.
    - etc...
5. The final goal is to detect the best grasp pose rectangle for the object (ONLY ONE) mentioned in the text prompt, do not return anything else except the grasp pose rectangle, do not return grasp pose rectangles for multiple objects.
6. Avoid making assumptions or guesses. If uncertain, use available skills to gather additional information.  

### You have the following skills:
Skill 1: Find objects by their names in the image, capable of detecting single or multiple objects at once. For example, "find apple" or "find bottle."
   -use function find(object_name: str). 
Skill 2: Find a part of an object by the object name and the part name e.g., "Find handle of the spoon", "Find tines of the fork."
   -use function find_part(object_name: str, part_name: str). 
Skill 3: Calculate the bounding box or grasp rectangle for a detected object based on the text prompt, e.g., "calculate the grasp rectangle for the detected apple."
   -use function grasp_detection(object_patch: ImagePatch)
Skill 4: Compare and sort objects by their position or attributes, e.g., "Find the leftmost object," "sort objects by size."
   -use function sort attribute of list of objects
Skill 5: Logical reasoning to refine grasp detection or analyze object relationships, e.g., "Determine if the grasp rectangle overlaps with another object."
   -use basic logical python code like math, if-else, etc.
Skill 6: Verify the properties of an object, e.g., "Verify if the object is black color," "Verify if the object is round shape."
   -use function verify_property(object_name: str, property: str)
Skill 7: Calculate the depth or distance of an object from the camera, e.g., "Calculate the distance of the apple from the camera."
   -Use function compute_depth:
Skill 8: Compute distance between two objects, e.g., "Calculate the distance between the apple and the banana."
   -Use simple math library to calculate the distance between two objects.
Skill 9: Answer to a basic question asked about the image, e.g., "What is the color of the apple?","What is this?"
   -Use function simple_query(question: str)
Skill 10: Check if the object exist in the image, e.g., "Is the boy in the image."
   -Use function exist(object_name: str)

### How to Use These Skills Ideally:
---
Example 1: "Grasp the handle of the mug."
Plans:
Step 1: Find the mug. Use the function find() with the object name "mug"
Step 2: Locate the handle of the mug. Use the function find_part() with the object name "mug" and the part name "handle."
Step 3: Calculate the grasp pose for the mug handle. Use the function grasp_detection() with the detected mug handle patch.

---
Example 2: "Grasp the apple on the plate."
Plans:
Step 1: Find apples in the image. Use the function find() with the object name "apple"
Step 2: Detect the plate. Use the function find() with the object name "plate."
Step 3: Identify which apple is on the plate. The apple on the plate is the one that have the center inside the bounding box of the plate.
Step 4: Calculate the grasp pose for the apple on the plate. Use the function grasp_detection() with the detected apple on the plate patch.

---
Example 4: "Grasp the top left toy in the image."
Plans:
Step 1: Find all toy in the image. Use the function find() with the object name "toy"
Step 2: Sort toy by their position. Use the function sort attribute of list of objects to sort the toy by horizontal_center attribute.
Step 3: Calculate the grasp pose for the top-left toy. Use the function grasp_detection() with the detected top-left toy patch.

---
Example 5: "Grasp the phone that is closest to the camera."
Plans:
Step 1: Find all phones in the image. Use the function find() with the object name "phone"
Step 2: Sort phone by distance from the camera. Use function sort and compute_depth to sort the phone by distance from the camera.
Step 4: Calculate the grasp pose for the closest phone. Use the function grasp_detection() with the detected closest phone patch.

----
Example 6: "Grasp the food box product on the front right side of the brown tissues box."
Plans:
Step 1: Find the food box product. Use the function find() with the object name "food box product"
step 2: Find the brown tissues box. Use the function find() with the object name "brown tissues box"
Step 3: Check the position of the food box product. Compare horizontal_center and vertical_center of the food box product with the brown tissues box to determine if the food box is on the front right side fo the brown tissue box.
Step 4: Calculate the grasp pose for the detected food box product. Use the function grasp_detection() with the detected food box product patch.

---
Example 7: "Grasp the blue and black pen."
Plans: 
Step 1: Find all pens in the image. Use the function find() with the object name "pen"
Step 2: Verify the color of the pens. Use the function verify_property() with the object name "pen" and the property "color" to check if the pen is both blue and black.
Step 3: Calculate the grasp pose for the detected blue and black pen. Use the function grasp_detection() with the detected blue and black pen patch.

## Please restrict the output follow the format plan in the above examples. Only generate the plan with the steps that are necessary to answer the question. The final step have to be the grasp pose calculation.
Format:
Step i: <Instruction i>. <Function/tool use to complete the instruction>.

Query is {query} 
'''

OUTPUT_FORMAT = '''
### Output a list of jsons in the following format:
```json
[
    {"id": int, "instruction": str, "probability": float},
    ...
]
```

Example Answer:  
For the question, "Grasp the handle of the mug."  
Step 1: No feedback available.  
Answer:  
```json
[
    {"id": 1, "instruction": "find mug", "probability": 0.8},
    {"id": 2, "instruction": "find handle of the mug", "probability": 0.9},
    {"id": 3, "instruction": "calculate grasp pose base on the detected handle of the mug", "probability": 0.85},
]
```  
'''


EXAMPLE = """
E.g., 
Question: "Grasp the handle of the mug."
Plan:
Step 1: Find mug
Step 2: Find handle of the mug
Step 3: Calculate the grasp rectangle for the detected mug handle

E.g.,
Question: "Grasp the apple on the plate."
Plan:
Step 1: Find apples
Step 2: Find plate
Step 3: Check each of the apples if they are on the plate, return the one that is on the plate
Step 4: Calculate the grasp rectangle for the detected apple on the plate

E.g.,
Question: "Give me the knife in safety."
Plan:
Step 1: Find knife
Step 2: In order to handover safety, find the blade of the knife to grasp, so the user could grasp the handle safely
Step 3: Calculate the grasp rectangle for the detected blade of the knife

E.g.,
Question: "Grasp the top left of an object."
Plan:
Step 1: Find objects
Step 2: Sort objects by position in vertical and horizontal directions, return the top left object
Step 3: Calculate the grasp rectangle for the detected top left object

E.g.,
Question: "Grasp the object that is closest to the camera."
Plan:
Step 1: Find objects
Step 2: Sort objects by distance from the camera, return the closest object
Step 3: Calculate the grasp rectangle for the detected closest object

---
Example 3: "Give me the knife in safety."

Step 1: Find the knife.  
   - Detect the knife in the image.
Step 2: Identify the blade of the knife for safe handling.  
   - To ensure safety, locate the blade of the knife, as this is the part to grasp, allowing the user to safely hold the handle.
Step 3: Calculate the grasp pose for the knife blade.  
   - Calculate the grasp pose for the detected knife blade to ensure a safe handover.
"""

