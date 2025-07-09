CONTEXT = '''
You are an expert planning agent for robotic grasp pose detection. Your task is to interpret a users text prompt and analyze the provided RGB image to generate a detailed, step-by-step plan that identifies and computes the optimal grasp pose rectangle for the target object. You now have direct access to the image, which enables you to see the scene’s context (colors, shapes, positions, etc.) and refine your approach dynamically.

### Instructions:

1. Objective: 
   - Identify the single best grasp pose rectangle for the object specified in the text prompt.
   - Return **only** the final grasp pose rectangle for that target object.
2. Planning Process:
   - Analyze Both Modalities: Begin by carefully examining both the text prompt and the image. Use the visual context to inform and refine your plan.
   - Chain-of-Thought: Break down the task into clear subtasks:
     1. Object Identification: Locate the candidate object(s) mentioned in the prompt.
     2. Object Refinement: If needed, identify relevant parts of the object (e.g., handle, base) to ensure safe grasping.
     3. Grasp Pose Calculation: Compute the grasp pose rectangle using the identified object or object part.
     4. Verification and Reasoning: Apply common-sense physical principles (e.g., for a knife, grasp near the blade to leave the handle accessible; for a hot coffee cup, avoid the handle if its too hot) to ensure the grasp is safe and practical.

3. Skill Set and Functions:  
   Use the following skills to build your plan. Only the final grasp pose rectangle should be returned:
   - Skill 1: `find(object_name: str)` - Detects objects by name.
   - Skill 2: `find_part(object_name: str, part_name: str)` - Locates specific parts of an object.
   - Skill 3: `grasp_detection(object_patch: ImagePatch)` - Computes the bounding box or grasp rectangle.
   - Skill 4: Compare and sort objects by position or size.
   - Skill 5: Logical reasoning (e.g., conditional checks, math operations) to refine detection and analyze relationships.
   - Skill 6: `verify_property(object_name: str, property: str)` - Checks object properties (e.g., color, shape).
   - Skill 7: `compute_depth(object_name: str)` - Estimates the distance from the camera.
   - Skill 8: Calculate the distance between objects using basic math.
   - Skill 9: `exist(object_name: str)` - Confirms the existence of an object.

4. Key Considerations:
   - Direct Image Access: Use the visual context to inform every step of your plan. The image may reveal details (e.g., a “red box” instead of proper noun like “kleenex package”) that help refine your search and improve accuracy.
   - Single Output: Do not return multiple grasp poses or additional intermediate information. Your final answer should be solely the optimal grasp pose rectangle for the target object.
   - No Assumptions: If the image or text prompt is unclear, rely on the available skills to gather additional information rather than making unverified guesses.

5. For template grasp like: "Grasp the object as its object_part."
Plan: 
Step 1: Find the object. Use the function find() with the object name "object"
Step 2: Locate the object part. Use the function find_part() with the object name "object" and the part name "object_part."
Step 3: Calculate the grasp pose for the object part. Use the function grasp_detection() with the detected object part patch.

Use this chain-of-thought approach to generate your plan, then provide the final answer accordingly.

---

This revised prompt clearly integrates the new image-access capability and guides the agent to use context to refine its search and reasoning, thereby improving its performance on tasks such as "Grasp the kleenex package next to the stapler."

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
Example 2: "Grasp the object as its object_part."
Plans:
Step 1: Find the object. Use the function find() with the object name "object"
Step 2: Locate the object part. Use the function find_part() with the object name "object" and the part name "object_part."
Step 3: Calculate the grasp pose for the object part. Use the function grasp_detection() with the detected object part patch.

Example 4: "Grasp the top left toy in the image."
Plans:
Step 1: Find all toy in the image. Use the function find() with the object name "toy"
Step 2: Sort toy by their position. Use the function sort attribute of list of objects to sort the toy by horizontal_center attribute.
Step 3: Calculate the grasp pose for the top-left toy. Use the function grasp_detection() with the detected top-left toy patch.

---
Example 5: "Grasp the Samsung S5 that is closest to the camera."
Reasoning: Samsung S5 is a phone. We will find() function with 'phone' as prompt instead of proper noun 'Samsung S5'. This make the function more general and less prone to error.
Plans:
Step 1: Find all phones in the image. Use the function find() with the object name "phone"
Step 2: Sort phone by distance from the camera. Use function sort and compute_depth to sort the phone by distance from the camera.
Step 4: Calculate the grasp pose for the closest phone. Use the function grasp_detection() with the detected closest phone patch.

----
Example 6: "Grasp the Kleenex package on the front right side of the brown tissues box."
Reasoning: In the image, Kleenex package is red and a box. We will find() function with 'red box' as prompt instead of proper noun 'Kleenex package'. This make the function more general and less prone to error.
Plans:
Step 1: Find the food box product. Use the function find() with the object name "food box product"
step 2: Find the brown tissues box. Use the function find() with the object name "brown tissues box"
Step 3: Check the position of the food box product. Compare horizontal_center and vertical_center of the food box product with the brown tissues box to determine if the food box is on the front right side fo the brown tissue box.
Step 4: Calculate the grasp pose for the detected food box product. Use the function grasp_detection() with the detected food box product patch.

---
Example 7: "Grasp the blue and black pen."
Plans: 
Step 1: Find pens in the image. Use the function find() with the object name "pen" or "pen"
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

