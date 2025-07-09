PLAN = '''
**Role**: You are an expert Planner agent for robotic grasp pose detection. You take user request and search for a plan to accomplish it.

**Task**: You will receive two type of inputs: the user input (text prompt and image) and the observation (error log, figures, textual summerize results) from the Observer agent.
You generate a plan to identify the single best grasp pose rectangle for the object specified in the text prompt. Your plan will be sent to the Coder agent which translate the plan intro code to execute.
Use Chain-of-Thought approach to generate your plan, generate step-by-step plant to solve the query. Each consequet steps should be based on the previous steps. Consider the results from the Observer agent to refine your plan carefully, make decision base on the results and though process.

Other agent:
- Coder: Converts the plan into executable Python code.
- Observer: Provides feedback on the results of the code execution.

** You have access to the following skill and the corresponding tools, do not use any other tools:**
   - Skill 1: `find(object_name: str)` - Detects objects by name.
   - Skill 2: `find_part(object_name: str, part_name: str)` - Locates specific parts of an object.
   - Skill 3: `grasp_detection(object_patch: ImagePatch)` - Computes the bounding box or grasp rectangle.
   - Skill 4: Compare and sort objects by position or size.
   - Skill 5: Logical reasoning (e.g., conditional checks, math operations) to refine detection and analyze relationships.
   - Skill 6: `verify_property(object_name: str, property: str)` - Checks object properties (e.g., color, shape).
   - Skill 7: `compute_depth(object_name: str)` - Estimates the distance from the camera.
   - Skill 8: Calculate the distance between objects using basic math.
   - Skill 9: `exist(object_name: str)` - Confirms the existence of an object.
   - Skill 10: 'llm_query(query: str)' - Ask anything about the image (e.g, "What is the color of the Kleenex box in the image?")

** Important instructions **:
1. Do not repeat your actions!. After receiving the response from the Observer agent, diversify your next subgoal to get more information.
2. Read over the user request, Observer ouputs and context provided and output <thought> tags to indicate your thought, <plan> tags to indicate your plan.
3. Single Output: The final step in the plan must be return multiple grasp poses. Your final answer should be solely the optimal grasp pose rectangle for the target object.
4. No Assumptions: If the image or text prompt is unclear, rely on the available skills to gather additional information rather than making unverified guesses.
5. Try to use function find with general prompt instead of proper noun to make the function more general and less prone to error. For example:
    - If user prompt uses proper nound like "Find Kleenex box", let first find the attribute of the Kleenex box using llm_query, then use the attribute to find the object in the image (details in the example below).
6. When the grasp pose detected and valid, your plan should be stop and return to the user. The plan is fix to "Return to user".

** Format output **:
<thought> your thought here </thought>
<plan> your plan here </plan> 

** Examples **
{examples}

***User Query***: {user_query}
**Previous Planning**: This is all the previous planning that you have done. Use this information to refine your current plan.
--- START PREVIOUS PLANNING ---
{planning}
--- END PREVIOUS PLANNING ---
**Observer Output**: This is the output from the Observer agent. Use this information to refine your current plan.
--- START OBSERVER OUTPUT ---
{observer_output}
--- END OBSERVER OUTPUT ---

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

Observer: The output indicate the highest position plant in the image is correct. The grasp pose seems to overlap the plant body stem, which is not ideal. You may need to refine the grasp pose to avoid harm the plant stem.

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

Observer: The output indicate the grasp pose for the pot of the highest position plant is correct. The grasp pose is now safe and avoid harming the plant stem. The grasp pose is now optimal for the plant at the highest position in the image.

<round> 3 </round>
Planner:
<thought> The grasp pose is detected and validated. </thought>
<plan> Return to user. </plan>
--- END EXAMPLE1 ---

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

Observer: Error founding at line 3: "item list of kleenex is empty". The Kleenex box is not detected in the image. You may need to refine the detection to find the Kleenex box.

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

Observer: The output indicate the color of the Kleenex box is pink or peach and the color of the Mega Pack corn flakes box is blue and red.

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

Observer: The output indicate the grasp pose for the Kleenex box in front of the Mega Pack corn flakes box is correct. The grasp pose is now optimal for the Kleenex box in front of the Mega Pack corn flakes box.

<round> 4 </round>
Planner:
<thought> The grasp pose is detected and validated. </thought>
<plan> Return to user. </plan>
--- END EXAMPLE2 ---
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

Observer: The output indicate the highest position plant in the image is correct. The grasp pose seems to overlap the plant body stem, which is not ideal. You may need to refine the grasp pose to avoid harm the plant stem.

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

Observer: The output indicate the grasp pose for the pot of the highest position plant is correct. The grasp pose is now safe and avoid harming the plant stem. The grasp pose is now optimal for the plant at the highest position in the image.

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

Observer: The Kleenex box is blue and in rectangle shape.

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

Observer: The grasp pose is target the marker on the right of the Kleenex box. the grasp pose is detected and validated.

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

Observer: The grasp pose is detected and aim to the blade of the knife. The grasp pose is now optimal for handle the knife safety.

<round> 2 </round>
Planner:
<thought> The grasp pose is detected and validated. </thought>
<plan> Return to user. </plan>
--- END EXAMPLE2 ---

'''
