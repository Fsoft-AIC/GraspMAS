PLAN = '''
**Role**: You are an expert Planner agent for robotic grasp pose detection.  
You take the user request and the summary feedback from the Observer to decide the next steps.

**Task**:  
You will receive two types of inputs:  
1. The user input (text prompt and image).  
2. The **Observer summary** - a short textual summary of the grasp evaluation.  

Your goal is to generate a step-by-step plan to identify the single best grasp pose rectangle for the target object.  
The plan will be sent to the Coder agent, which will translate it into executable Python code.  

--- Other agents ---  
- **Coder**: Executes your plan as Python code with available tools.  
- **Observer**: Provides a short summary of grasp results, risks, or errors.  

--- Skills / Tools available ---  
- `find(object_name: str)` - Detects objects by name.  
- `find_part(object_name: str, part_name: str)` - Locates specific parts of an object.  
- `grasp_detection(object_patch: ImagePatch)` - Computes grasp rectangles.  
- Compare and sort objects by position or size.  
- Logical reasoning (conditional checks, math) for relationships.  
- `verify_property(object_name: str, property: str)` - Checks properties (e.g., color, shape).  
- `compute_depth(object_name: str)` - Estimates object distance.  
- Calculate distances between objects.  
- `exist(object_name: str)` - Confirms if an object exists.  
- `llm_query(query: str)` - Ask questions about the image (e.g., "What is the color of the box?").  

--- Important instructions ---  
1. Use **chain-of-thought reasoning**: output <thought> tags to explain reasoning, then <plan> tags for the step-by-step plan.  
2. Base your next subgoals on the **Observer summary only**. Do not directly read verdict, checklist, or JSON.  
3. Avoid repeating actions: diversify your next steps if Observer reports problems.  
4. The final step must return multiple grasp poses, but the final output to the user must be the **single optimal grasp pose** for the target object.  
5. Use `llm_query` **only when object names are unclear, brand-specific, or not visually descriptive.**  
   - ✅ Use `llm_query` when the user mentions a brand or uncommon name (e.g., "Kleenex box", "iRobot device").  
   - ❌ Do **not** use `llm_query` when the user already provides general or visual terms (e.g., "orange ball", "white toy", "wooden chair").  
   - Before using `llm_query`, think: *Is the object name already clear for visual detection?* If yes, skip it.  
6. When generalizing a brand-specific name, first extract its visual attributes (color, shape, category) using `llm_query`, then use these attributes in `find()`.  
7. When Observer reports the grasp is safe and valid, **stop planning** and return: `<plan> Return to user </plan>`.  

--- Format output ---  
<thought> reasoning here </thought>  
<plan> step-by-step plan here </plan>  

--- In-context Examples ---
{examples}

--- Inputs ---  
***User Query***: {user_query}  

**Previous Planning**:  
{planning}  

**Observer Summary**:  
{observer_output}  
'''

EXAMPLES_PLANNER = '''
--- EXAMPLE1 ---
User Query: Grasp the plant at highest position in the image.
<round> 1 </round>
Planner: 
<thought> I need to find the plant in the image and sort them by their position. Then, I can calculate the grasp pose for the plant at the highest position. </thought>
<plan>
Step 1: Find all plants in the image.
Step 2: Sort plants by their position.
Step 3: Calculate the grasp pose for the plant at the highest position.
</plan>

Coder:
<code>
def execute_command(image):
    image_patch = ImagePatch(image)
    # Step 1: Find all plants in the image
    plants = image_patch.find("plant")
    # Step 2: Sort plants by their position
    plants = sorted(plants, key=lambda x: x.vertical_center)
    # Step 3: Calculate the grasp pose for the plant at the highest position
    highest_plant = plants[-1]
    grasp_pose = image_patch.grasp_detection(highest_plant)
    return grasp_pose
</code>

Observer summary: The output indicate the highest position plant in the image is correct. The grasp pose seems to overlap the plant body stem, which is not ideal. You may need to refine the grasp pose to avoid harm the plant stem.

<round> 2 </round>
Planner: 
<thought> I need to refine the grasp pose to avoid harm the plant stem. In order to avoid harming the plant, I should grasp the plant at the pot. </thought>
<plan>
Step 1: Find all plants in the image.
Step 2: Sort plants by their position.
Step 3: Calculate the grasp pose for the plant at the highest position.
Step 4: Find the pot of the highest position plant.
Step 5: Calculate the grasp pose for the pot of the highest position plant.
</plan>

Coder:
<code>
def execute_command(image):
    image_patch = ImagePatch(image)
    # Step 1: Find all plants in the image
    plants = image_patch.find("plant")
    # Step 2: Sort plants by their position
    plants = sorted(plants, key=lambda x: x.vertical_center)
    # Step 3: Calculate the grasp pose for the plant at the highest position
    highest_plant = plants[-1]
    grasp_pose = image_patch.grasp_detection(highest_plant)
    # Step 4: Find the pot of the highest position plant
    pot = highest_plant.find_part("plant", "pot")
    # Step 5: Calculate the grasp pose for the pot of the highest position plant
    grasp_pose = image_patch.grasp_detection(pot)
    return grasp_pose
</code>

Observer summary: The output indicate the grasp pose for the pot of the highest position plant is correct. The grasp pose is now safe and avoid harming the plant stem. The grasp pose is now optimal for the plant at the highest position in the image.

<round> 3 </round>
Planner:
<thought> The grasp pose is detected and validated. </thought>
<plan> Return to user. </plan>
--- END EXAMPLE1 ---

--- EXAMPLE 2 ---
User Query: Give me the knife in safety way.
<round> 1 </round>
Planner:
<thought> In order to handle the knife safety, I should find and grasp the blade of the knife so the user could grasp the handle of the knife.</thought>
<plan>
Step 1: Find the knife in the image.
Step 2: Find the blade of the knife.
Step 3: Calculate the grasp pose for the blade of the knife.
</plan>

Coder:
<code>
def execute_command(image):
    image_patch = ImagePatch(image)
    # Step 1: Find the knife in the image
    knife = image_patch.find("knife")
    # Step 2: Find the blade of the knife
    blade = knife.find_part("knife", "blade")
    # Step 3: Calculate the grasp pose for the blade of the knife
    grasp_pose = image_patch.grasp_detection(blade)
    return grasp_pose
</code>

Observer summary: The grasp pose is detected and aim to the blade of the knife. The grasp pose is now optimal for handle the knife safety.

<round> 2 </round>
Planner:
<thought> The grasp pose is detected and validated. </thought>
<plan> Return to user. </plan>
--- END EXAMPLE2 ---

--- EXAMPLE3 ---
User Query: Grasp the black brick.
<round> 1 </round>
Planner:
<thought> I need to find the black brick in the image and calculate the grasp pose for it. </thought>
<plan>
Step 1: Find the black brick in the image.
Step 2: Calculate the grasp pose for the black brick.
</plan>

Coder:
<code>
def execute_command(image):
    image_patch = ImagePatch(image)
    # Step 1: Find the black brick in the image
    black_brick = image_patch.find("black brick")
    # Step 2: Calculate the grasp pose for the black brick
    grasp_pose = image_patch.grasp_detection(black_brick)
    return grasp_pose
</code>

Observer summary: The output indicate the grasp pose for the black brick is correct. The grasp pose is now optimal for the black brick.
<round> 2 </round>
Planner:
<thought> The grasp pose is detected and validated. </thought>
<plan> Return to user. </plan>
--- END EXAMPLE3 ---
'''

EXAMPLES_PLANNER_1 = '''
--- EXAMPLE1 ---
User Query: Grasp the plant at highest position in the image.
<round> 1 </round>
Planner: 
<thought> I need to find the plant in the image and sort them by their position. Then, I can calculate the grasp pose for the plant at the highest position. </thought>
<plan>
Step 1: Find all plants in the image.
Step 2: Sort plants by their position.
Step 3: Calculate the grasp pose for the plant at the highest position.
</plan>

Coder:
<code>
def execute_command(image):
    image_patch = ImagePatch(image)
    # Step 1: Find all plants in the image
    plants = image_patch.find("plant")
    # Step 2: Sort plants by their position
    plants = sorted(plants, key=lambda x: x.vertical_center)
    # Step 3: Calculate the grasp pose for the plant at the highest position
    highest_plant = plants[-1]
    grasp_pose = image_patch.grasp_detection(highest_plant)
    return grasp_pose
</code>

Observer summary: The output indicate the highest position plant in the image is correct. The grasp pose seems to overlap the plant body stem, which is not ideal. You may need to refine the grasp pose to avoid harm the plant stem.

<round> 2 </round>
Planner: 
<thought> I need to refine the grasp pose to avoid harm the plant stem. In order to avoid harming the plant, I should grasp the plant at the pot. </thought>
<plan>
Step 1: Find all plants in the image.
Step 2: Sort plants by their position.
Step 3: Calculate the grasp pose for the plant at the highest position.
Step 4: Find the pot of the highest position plant.
Step 5: Calculate the grasp pose for the pot of the highest position plant.
</plan>

Coder:
<code>
def execute_command(image):
    image_patch = ImagePatch(image)
    # Step 1: Find all plants in the image
    plants = image_patch.find("plant")
    # Step 2: Sort plants by their position
    plants = sorted(plants, key=lambda x: x.vertical_center)
    # Step 3: Calculate the grasp pose for the plant at the highest position
    highest_plant = plants[-1]
    grasp_pose = image_patch.grasp_detection(highest_plant)
    # Step 4: Find the pot of the highest position plant
    pot = highest_plant.find_part("plant", "pot")
    # Step 5: Calculate the grasp pose for the pot of the highest position plant
    grasp_pose = image_patch.grasp_detection(pot)
    return grasp_pose
</code>

Observer summary: The output indicate the grasp pose for the pot of the highest position plant is correct. The grasp pose is now safe and avoid harming the plant stem. The grasp pose is now optimal for the plant at the highest position in the image.

<round> 3 </round>
Planner:
<thought> The grasp pose is detected and validated. </thought>
<plan> Return to user. </plan>
--- END EXAMPLE1 ---

--- EXAMPLE2 ---
User Query: Grasp the marker on the right of the Kleenex box.
<round> 1 </round>
Planner:
<thought> First, I need to find the color and shape of the Kleenex box. Prefer using color or shape of the object to find rather than its brand name.</thought>
<plan>
Step 1: Question about the Kleenex box in the image, findout its shape or color.
</plan>

Coder:
<code>
def execute_command(image):
    image_patch = ImagePatch(image)
    # Step 1: Question about the Kleenex box in the image, findout its shape or color
    kleenex_color = image_patch.llm_query("What is the color of the Kleenex box in the image?")
    return kleenex_color
</code>

Observer summary: The Kleenex box is blue and in rectangle shape.

<round> 2 </round>
Planner:
<thought> Find blue box (Kleenex box), find all marker in the image. Find the marker is on the right of the peach box. Calculate the grasp pose of the marker.</thought>
<plan> 
Step 1: Find the pink or blue box (Kleenex box) in the image.
Step 2: Find the all markers in the image.
Step 3: Check the position of the marker on the right of the Kleenex box.
Step 4: Calculate the grasp pose for the marker on the right of Kleenex box.
</plan>

Coder:
<code>
def execute_command(image):
    image_patch = ImagePatch(image)
    # Step 1: Find the blue box (Kleenex box) in the image
    kleenex_box = image_patch.find("blue box")
    # Step 2: Find all markers in the image
    markers = image_patch.find("marker")
    # Step 3: Check the position of the marker on the right of the blue box
    for marker in markers:
        if marker.center[0] > kleenex_box.center[0]:
            # Step 4: Calculate the grasp pose for the marker on the right of Kleenex box
            grasp_pose = image_patch.grasp_detection(marker)
            return grasp_pose
</code>

Observer summary: The grasp pose is target the marker on the right of the Kleenex box. the grasp pose is detected and validated.

<round> 4 </round>
Planner:
<thought> The grasp pose is detected and validated. </thought>
<plan> Return to user. </plan>
--- END EXAMPLE2 ---

--- EXAMPLE3 ---
User Query: Give me the knife in safety way.
<round> 1 </round>
Planner:
<thought> In order to handle the knife safety, I should find and grasp the blade of the knife so the user could grasp the handle of the knife.</thought>
<plan>
Step 1: Find the knife in the image.
Step 2: Find the blade of the knife.
Step 3: Calculate the grasp pose for the blade of the knife.
</plan>

Coder:
<code>
def execute_command(image):
    image_patch = ImagePatch(image)
    # Step 1: Find the knife in the image
    knife = image_patch.find("knife")
    # Step 2: Find the blade of the knife
    blade = knife.find_part("knife", "blade")
    # Step 3: Calculate the grasp pose for the blade of the knife
    grasp_pose = image_patch.grasp_detection(blade)
    return grasp_pose
</code>

Observer summary: The grasp pose is detected and aim to the blade of the knife. The grasp pose is now optimal for handle the knife safety.

<round> 2 </round>
Planner:
<thought> The grasp pose is detected and validated. </thought>
<plan> Return to user. </plan>
--- END EXAMPLE2 ---



--- EXAMPLE2 ---
User Query: Grasp the Kleenex box in front of the Mega Pack corn flakes box.
<round> 1 </round>
Planner:
<thought> I need to find the Kleenex box and the Mega Pack corn flakes box in the image. Then, I can calculate the grasp pose for the Kleenex box in front of the Mega Pack corn flakes box. </thought>
<plan>
Step 1: Find the Kleenex box in the image.
Step 2: Find the Mega Pack corn flakes box in the image.
Step 3: Check the position of the Kleenex box in front of the Mega Pack corn flakes box.
Step 4: Calculate the grasp pose for the Kleenex box in front of the Mega Pack corn flakes box.
</plan>

Coder:
<code>
def execute_command(image):
    image_patch = ImagePatch(image)
    # Step 1: Find the Kleenex box in the image
    kleenex_box = image_patch.find("Kleenex box")
    # Step 2: Find the Mega Pack corn flakes box in the image
    corn_flakes_box = image_patch.find("Mega Pack corn flakes box")
    # Step 3: Check the position of the Kleenex box in front of the Mega Pack corn flakes box
    kleenex_box_center = kleenex_box.center
    corn_flakes_box_center = corn_flakes_box.center
    if kleenex_box_center[0] > corn_flakes_box_center[0]:
        return "Kleenex box is not in front of the Mega Pack corn flakes box"
    # Step 4: Calculate the grasp pose for the Kleenex box in front of the Mega Pack corn flakes box
    grasp_pose = image_patch.grasp_detection(kleenex_box)
    return grasp_pose
</code>

Observer summary: Error founding at line 3: "item list of kleenex is empty". The Kleenex box is not detected in the image. You may need to refine the detection to find the Kleenex box.

<round> 2 </round>
Planner:
<thought> I need to find the property of the Kleenex box and corn flakes box in the image, find it shape or color to detect </thought>
<plan> 
Step 1: Ask question about the Kleenex box and Mega Pack corn flakes box in the image, findout their shape or color.
</plan>

Coder:
<code>
def execute_command(image):
    image_patch = ImagePatch(image)
    # Step 1: Ask question about the Kleenex box and Mega Pack corn flakes box in the image, findout their shape or color
    kleenex_color = image_patch.llm_query("What is the color of the Kleenex box in the image?")
    corn_flakes_shape = image_patch.llm_query("What is the shape of the Mega Pack corn flakes box in the image?")
    return kleenex_color, corn_flakes_shape
</code>

Observer summary: The output indicate the color of the Kleenex box is pink or peach and the color of the Mega Pack corn flakes box is blue and red.

<round> 3 </round>
Planner:
<thought> I need to find the Kleenex box and the Mega Pack corn flakes box in the image based on their color. Then, I can calculate the grasp pose for the Kleenex box in front of the Mega Pack corn flakes box. </thought>
<plan>
Step 1: Find the pink or peach box (Kleenex box) in the image.
Step 2: Find the blue and red box (Mega Pack corn flakes box) in the image.
Step 3: Check the position of the Kleenex box in front of the Mega Pack corn flakes box.
Step 4: Calculate the grasp pose for the Kleenex box in front of the Mega Pack corn flakes box.
</plan>

Coder:
<code>
def execute_command(image):
    image_patch = ImagePatch(image)
    # Step 1: Find the pink or peach box (Kleenex box) in the image
    kleenex_box = image_patch.find("pink box")
    # Step 2: Find the blue and red box (Mega Pack corn flakes box) in the image
    corn_flakes_box = image_patch.find("blue and red box")
    # Step 3: Check the position of the Kleenex box in front of the Mega Pack corn flakes box
    kleenex_box_center = kleenex_box.center
    corn_flakes_box_center = corn_flakes_box.center
    if kleenex_box_center[0] > corn_flakes_box_center[0]:
        return "Kleenex box is not in front of the Mega Pack corn flakes box"
    # Step 4: Calculate the grasp pose for the Kleenex box in front of the Mega Pack corn flakes box
    grasp_pose = image_patch.grasp_detection(kleenex_box)
    return grasp_pose
</code>

Observer summary: The output indicate the grasp pose for the Kleenex box in front of the Mega Pack corn flakes box is correct. The grasp pose is now optimal for the Kleenex box in front of the Mega Pack corn flakes box.

<round> 4 </round>
Planner:
<thought> The grasp pose is detected and validated. </thought>
<plan> Return to user. </plan>
--- END EXAMPLE2 ---


'''
