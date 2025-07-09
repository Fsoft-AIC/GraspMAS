context = '''
You are a software programmer. Use a Chain-of-Thought approach to break down the problem, create pseudocode, and then write the code in Python language. Ensure that your code is efficient, readable, and well-commented. Return the requested information from the function you create.
Setting:
    1. You will be given the instructions to identify a feasible grasp pose using RGB input. Please use the class python below to write the python code base on the instructions.
    2. Your primary responsibility is to translate instructions into Python code. This code will aid in obtaining more visual perception information and conducting logical analysis to arrive at the final answer for query.
    3. Image patch is a crop of an image centered around a particular object.
    4. You can use base Python (comparison) for basic logical operations, math, etc.

Provided Python Functions/Class:

import math
class ImagePatch:
    """A Python class containing a crop of an image centered around a particular object, as well as relevant information.
    Attributes
    ----------
    cropped_image : array_like
        An array-like of the cropped image taken from the original image.
    left : int
        An int describing the position of the left border of the crop's bounding box in the original image.
    lower : int
        An int describing the position of the bottom border of the crop's bounding box in the original image.
    right : int
        An int describing the position of the right border of the crop's bounding box in the original image.
    upper : int
        An int describing the position of the top border of the crop's bounding box in the original image.
    vertical_center: int
        An int describing the vertical center of the crop's bounding box in the original image. Higher values are closer to the top.
    horizontal_center: int
        An int describing the horizontal center of the crop's bounding box in the original image. Higher values are closer to the right.

    Methods
    -------
    find(object_name: str)->list[ImagePatch]
        Returns a list of new ImagePatch objects containing crops of the image centered around any objects found in the
        image matching the object_name.
    exists(object_name: str)->bool
        Returns True if the object specified by object_name is found in the image, and False otherwise.
    verify_property(property: str)->bool
        Returns True if the property is met, and False otherwise.
    best_text_match(option_list: list[str], prefix: str)->str
        Returns the string that best matches the image.
    simple_query(question: str=None)->str
        Returns the answer to a basic question asked about the image. If no question is provided, returns the answer to "What is this?".
    llm_query(question: str, object_name:str, long_answer: bool)->str
        References a large language model (e.g., GPT) to produce a response to the given question. Default is short-form answers, can be made long-form responses with the long_answer flag.
    compute_depth()->float
        Returns the median depth of the image crop.
    crop(left: int, lower: int, right: int, upper: int)->ImagePatch
        Returns a new ImagePatch object containing a crop of the image at the given coordinates.
    find_part(object_name: str, part_name: str)->ImagePatch
        Returns a new ImagePatch object containing crops of the image centered around a part of an object (object_name) in the image that
        matching the part_name 
    grasp_detection(object_patch: ImagePatch):->List[float]
        Return a best grasp pose detected with given object_patch object (contain mask and image of crops of object part)
    """

    def __init__(self, image, left: int = None, lower: int = None, right: int = None, upper: int = None):
        """Initializes an ImagePatch object by cropping the image at the given coordinates and stores the coordinates as
        attributes. If no coordinates are provided, the image is left unmodified, and the coordinates are set to the
        dimensions of the image.
        Parameters
        -------
        image : array_like
            An array-like of the original image.
        left, lower, right, upper : int
            An int describing the position of the (left/lower/right/upper) border of the crop's bounding box in the original image.
        """
        if left is None and right is None and upper is None and lower is None:
            self.cropped_image = image
            self.left = 0
            self.lower = 0
            self.right = image.shape[2]  # width
            self.upper = image.shape[1]  # height
        else:
            self.cropped_image = image[:, lower:upper, left:right]
            self.left = left
            self.upper = upper
            self.right = right
            self.lower = lower

        self.width = self.cropped_image.shape[2]
        self.height = self.cropped_image.shape[1]

        self.horizontal_center = (self.left + self.right) / 2
        self.vertical_center = (self.lower + self.upper) / 2

    def find(self, object_name: str) -> list[ImagePatch]:
        """Returns a list of ImagePatch objects matching object_name contained in the crop if any are found.
        Otherwise, returns an empty list.
        Parameters
        ----------
        object_name : str
            the name of the object to be found

        Returns
        -------
        list[ImagePatch]
            a list of ImagePatch objects matching object_name contained in the crop

        Examples
        --------
        >>> # return the foo
        >>> def execute_command(image) -> list[ImagePatch]:
        >>>    image_patch = ImagePatch(image)
        >>>    foo_patches = image_patch.find("foo")
        >>>    return foo_patches

        >>> # Generate the mask of the green book
        >>> def execute_command(image) -> str:
        >>>    image_patch = ImagePatch(image)
        >>>    book_patches = image_patch.find("book")
        >>>    for book_patch in book_patches:
        >>>        if book_patch.verify_property("book", "green"):
        >>>            return book_patch.mask
        
        >>> # Which orange is the leftmost
        >>> def execute_command(image) -> str:
        >>>    image_patch = ImagePatch(image)
        >>>    orange_patches = image_patch.find("orange")
        >>>    orange_patches.sort(key=lambda x: x.left)
        >>>    return orange_patches[0].left, orange_patches[0].lower, orange_patches[0].right, orange_patches[0].upper
        
        >>> # Generate the bounding boxes for all dogs
        >>> def execute_command(image) -> str:
        >>>    image_patch = ImagePatch(image)
        >>>    dog_patches = image_patch.find("dog")
        >>>    bounding_boxes = []
        >>>    for dog_patch in dog_patches:
        >>>        if dog_patch.exists("dog"):
        >>>        bounding_boxes.append(dog_patch.left, dog_patch.lower, dog_patch.right, dog_patch.upper)
        >>>     return bounding_boxes
        """
        return find_in_image(self.cropped_image, object_name)

    def exists(self, object_name: str) -> bool:
        """Returns True if the object specified by object_name is found in the image, and False otherwise.
        Parameters
        -------
        object_name : str
            A string describing the name of the object to be found in the image.

        Examples
        -------
        >>> # Are there both foos and garply bars in the photo?
        >>> def execute_command(image)->str:
        >>>     image_patch = ImagePatch(image)
        >>>     is_foo = image_patch.exists("foo")
        >>>     is_garply_bar = image_patch.exists("garply bar")
        >>>     return image_patch.bool_to_yesno(is_foo and is_garply_bar)
        """
        return len(self.find(object_name)) > 0

    def verify_property(self, object_name: str, visual_property: str) -> bool:
        """Returns True if the object possesses the visual property, and False otherwise.
        Differs from 'exists' in that it presupposes the existence of the object specified by object_name, instead checking whether the object possesses the property.
        Parameters
        -------
        object_name : str
            A string describing the name of the object to be found in the image.
        visual_property : str
            A string describing the simple visual property (e.g., color, shape, material) to be checked.

        Examples
        -------
        >>> # Do the letters have blue color?
        >>> def execute_command(image) -> str:
        >>>     image_patch = ImagePatch(image)
        >>>     letters_patches = image_patch.find("letters")
        >>>     # Question assumes only one letter patch
        >>>     return image_patch.bool_to_yesno(letters_patches[0].verify_property("letters", "blue"))
        """
        return verify_property(self.cropped_image, object_name, property)

    def simple_query(self, question: str = None) -> str:
        """Returns the answer to a basic question asked about the image. If no question is provided, returns the answer
        to "What is this?". The questions are about basic perception, and are not meant to be used for complex reasoning
        or external knowledge.
        Parameters
        -------
        question : str
            A string describing the question to be asked.

        Examples
        -------

        >>> # Which kind of baz is not fredding?
        >>> def execute_command(image) -> str:
        >>>     image_patch = ImagePatch(image)
        >>>     baz_patches = image_patch.find("baz")
        >>>     for baz_patch in baz_patches:
        >>>         if not baz_patch.verify_property("baz", "fredding"):
        >>>             return baz_patch.simple_query("What is this baz?")

        >>> # What color is the foo?
        >>> def execute_command(image) -> str:
        >>>     image_patch = ImagePatch(image)
        >>>     foo_patches = image_patch.find("foo")
        >>>     foo_patch = foo_patches[0]
        >>>     return foo_patch.simple_query("What is the color?")

        >>> # Is the second bar from the left quuxy?
        >>> def execute_command(image) -> str:
        >>>     image_patch = ImagePatch(image)
        >>>     bar_patches = image_patch.find("bar")
        >>>     bar_patches.sort(key=lambda x: x.horizontal_center)
        >>>     bar_patch = bar_patches[1]
        >>>     return bar_patch.simple_query("Is the bar quuxy?")
           
        """
        return simple_query(self.cropped_image, question)

    def crop(self, left: int, lower: int, right: int, upper: int, mask) -> ImagePatch:
        """Returns a new ImagePatch cropped from the current ImagePatch.
        Parameters
        -------
        left, lower, right, upper : int
            The (left/lower/right/upper)most pixel of the cropped image.
        mask
            A mask of the the most prominent object in the crop region. 
        """
        return ImagePatch(self.cropped_image, left, lower, right, upper, mask)

    def best_image_match(list_patches: list[ImagePatch], content: list[str], return_index=False) -> Union[ImagePatch, int]:
        """Returns the patch most likely to contain the content.
        Parameters
        ----------
        list_patches : list[ImagePatch]
        content : list[str]
            the object of interest
        return_index : bool
            if True, returns the index of the patch most likely to contain the object

        Returns
        -------
        int
            Patch most likely to contain the object
        """
        return best_image_match(list_patches, content, return_index)
        
    def compute_depth(self):
        """Returns the median depth of the image crop
        Parameters
        ----------
        Returns
        -------
        float
            the median depth of the image crop

        Examples
        --------
        >>> # the bar furthest away
        >>> def execute_command(image)->ImagePatch:
        >>>     image_patch = ImagePatch(image)
        >>>     bar_patches = image_patch.find("bar")
        >>>     bar_patches.sort(key=lambda bar: bar.compute_depth())
        >>>     return bar_patches[-1]
        """
        depth_map = compute_depth(self.cropped_image)
        return depth_map.median()
        
    def find_part(self, object_name: str, part_name: str) -> ImagePatch:
        """Returns a new ImagePatch object containing crops of the image centered around a part of an object (object_name) in the image that
        matching the part_name 
        """
        Parameters
        ----------
        object_name : str
            the object of interest
        part_name : str
            the part of the object of interest

        Returns
        -------
        ImagePatch
            ImagePatch of the part of the object of interest

        Examples
        --------
        >>> # Find the blade of the knife
        >>> def execute_command(image)->ImagePatch:
        >>>     image_patch = ImagePatch(image)
        >>>     knife_patches = image_patch.find("knife")
        >>>     knife_blade_patch = knife_patch[0].find_part("knife", "blade")
        >>>     return knife_blade_patch
        >>>
        >>> # Return the mask of the handle of the spoon
        >>> def execute_command(image)->ImagePatch
        >>>     image_patch = ImagePatch(image)
        >>>     spoon_patches = image_patch.find("spoon")
        >>>     spoon_handle_patch = spoon_patch[0].find_part("spoon", "handle")
        >>>     return spoon_handle_patch.mask
        """
        return find_part(object_name, part_name)
        
    def grasp_detection(object_patch: ImagePatch)->List[float]:
        """Returns the grasp pose of the object/part of object that centered in the object_patch
        Parameters
        ----------
        object_patch : ImagePatch
            the object of interest

        Returns
        -------
        List[float]
            the grasp pose of the object/part of object that centered in the object_patch
            
        Examples
        --------
        >>> # Return the grasp pose of the object
        >>> def execute_command(image):
        >>>     image_patch = ImagePatch(image)
        >>>     object_patches = image_patch.find("object")
        >>>     grasp_pose = image_patch.grasp_detection(object_patch)
        >>>     return grasp_pose
        >>>
        >>> # Give me the glasses
        >>> def execute_command(image):
        >>>     image_patch = ImagePatch(image)
        >>>     glasses_patches = image_patch.find("glasses")
        >>>     grasp_pose = image_patch.grasp_detection(glasses_patches[0])
        >>>     return grasp_pose
        >>>
        >>> # Grasp the plant at its pot
        >>> def execute_command(image):
        >>>     image_patch = ImagePatch(image)
        >>>     plant_patches = image_patch.find("plant")
        >>>     pot_patches = plant_patches[0].find_part("plant", "pot")
        >>>     grasp_pose = image_patch.grasp_detection(pot_patches[0])
        >>>     return grasp_pose
        >>>
        >>> # Grasp the orange on the plate
        >>> def execute_command(image):
        >>>     image_patch = ImagePatch(image)
        >>>     orange_patches = image_patch.find("orange")
        >>>     plate_patches = image_patch.find("plate")[0]
        >>>     oranges_on_plate = [
        >>>         orange for orange in orange_patches 
        >>>        if ( orange.vertical_center > plate_patches.lower and orange.vertical_center < plate_patches.upper and 
        >>>        orange.horizontal_center > plate_patches.left and orange.horizontal_center < plate_patches.right )
        >>>     ]
        >>>     if len(oranges_on_plate) == 0:
        >>>         return None
        >>>     grasp_pose = image_patch.grasp_detection(oranges_on_plate[0])
        >>>     return grasp_pose
        >>>
        >>> # Grasp the furtherest object
        >>> def execute_command(image):
        >>>     image_patch = ImagePatch(image)
        >>>     object_patches = image_patch.find("object")
        >>>     object_patches.sort(key=lambda object: object.compute_depth())
        >>>     grasp_pose = image_patch.grasp_detection(object_patches[-1])
        >>>     return grasp_pose
        >>>
        >>> # Grasp the knife next to the fork
        >>> def execute_command(image):
        >>>     image_patch = ImagePatch(image)
        >>>     fork_patches = image_patch.find("fork")
        >>>     knife_patches = image_patch.find("knife")
        >>>     knife_patches.sort(key=lambda knife: abs(knife.horizontal_center - fork_patches[0].horizontal_center))
        >>>     grasp_pose = image_patch.grasp_detection(knife_patches[0])
        >>>     return grasp_pose
        """
        return grasp_detection(object_patch)

    def overlaps_with(self, left, lower, right, upper):
        """Returns True if a crop with the given coordinates overlaps with this one,
        else False.
        Parameters
        ----------
        left, lower, right, upper : int
            the (left/lower/right/upper) border of the crop to be checked

        Returns
        -------
        bool
            True if a crop with the given coordinates overlaps with this one, else False

        Examples
        --------
        >>> # black foo on top of the qux
        >>> def execute_command(image) -> ImagePatch:
        >>>     image_patch = ImagePatch(image)
        >>>     qux_patches = image_patch.find("qux")
        >>>     qux_patch = qux_patches[0]
        >>>     foo_patches = image_patch.find("black foo")
        >>>     for foo in foo_patches:
        >>>         if foo.vertical_center > qux_patch.vertical_center
        >>>             return foo
        """
        return self.left <= right and self.right >= left and self.lower <= upper and self.upper >= lower


Write a function using Python and the ImagePatch class (above) that could be executed to provide an answer to the query. 

Consider the following guidelines:
    Objective:
    - Your primary goal is to identify a feasible grasp pose for a given object using RGB and depth inputs. You must ensure that the solution adheres to physical common sense and avoids harming people or objects.
    - Operate with safety as a top priority. Avoid suggesting actions that could damage the object, the environment, or harm humans.
    - Only generate code
    Programming Constraints:
    - Use base Python (comparison, sorting) for basic logical operations, left/right/up/down, math, etc.

Query: {query}
Plan: {plan}
def execute_command(image):
'''

__context = '''
### Example 1
Q: Write a python code to solve the following coding problem that obeys the constraints. The output code needs to {FEW_SHOT_QUESTION_GUIDE}. Please wrap your code answer using ```:
Query: return the foo
Plan: 
Step 1: Find foo
A: ```
def execute_command(image):
    image_patch = ImagePatch(image)
    foo_patches = image_patch.find("foo")
    return foo_patches
```

### Example 2
Q: Write a python code to solve the following coding problem that obeys the constraints. The output code needs to {FEW_SHOT_QUESTION_GUIDE}. Please wrap your code answer using ```:           
Query: Generate the mask of the black clock
Plan:
Step 1: Locate all instances of "clock" in the image.
    -Search for all patches or regions containing the object "clock" in the image.
Step 2: Check the color property of each detected clock.
    -For each detected "clock," verify if the color property matches the condition (e.g., "black").
Step 3: Return the mask of the first clock that meets the condition.
    -If a clock with the specified color is found, return its mask.
Step 4: Return None if no clock meets the condition.
    -If no clock meets the color condition, return None to indicate no suitable clock was found.
A: ```
def execute_command(image):
    image_patch = ImagePatch(image)
    clock_patches = image_patch.find("clock")
    for clock_patch in clock_patches:
        if clock_patch.verify_property("clock", "black"):
            return clock_patch.mask
    return None
    ```

### Example 3
Q: Write a python code to solve the following coding problem that obeys the constraints. The output code needs to {FEW_SHOT_QUESTION_GUIDE}. Please wrap your code answer using ```:          
Query: Generate the bounding boxes for all dogs
Plan: 
Step 1: Find all instances of "dog" in the image.
    -Search for all patches or regions containing the object "dog" in the image.
Step 2: Initialize an empty list to store bounding box coordinates.
    -Prepare a list to collect the bounding box coordinates for each detected dog.
Step 3: Verify each detected patch to ensure it contains a dog.
    -For each detected patch, check if it is indeed a dog (using the exists function).
Step 4: Extract and store the bounding box coordinates for each valid dog.
    -If the patch contains a dog, extract its bounding box coordinates (left, lower, right, upper) and store them in the list.
Step 5: Return the list of bounding boxes for all valid dogs.
    -After processing all patches, return the list containing the bounding boxes for all valid dogs.
A: ```
def execute_command(image):
    image_patch = ImagePatch(image)
    dog_patches = image_patch.find("dog")
    bounding_boxes = []
    for dog_patch in dog_patches:
        if dog_patch.exists("dog"):
        bounding_boxes.append(dog_patch.left, dog_patch.lower, dog_patch.right, dog_patch.upper)
    return bounding_boxes
    ```

### Example 4
Q: Write a python code to solve the following coding problem that obeys the constraints. The output code needs to {FEW_SHOT_QUESTION_GUIDE}. Please wrap your code answer using ```: 
Query: Are there both foos and garply bars in the photo?
Plan: 
Step 1: Check if "foo" exists in the image.
Step 2: Check if "garply bar" exists in the image.
Step 3: Combine the results using a logical AND operation.
Step 4: Convert the boolean result to a "yes" or "no" response.
Step 5: Return the "yes" or "no" response.
A: ```
def execute_command(image):
    image_patch = ImagePatch(image)
    is_foo = image_patch.exists("foo")
    is_garply_bar = image_patch.exists("garply bar")
    return image_patch.bool_to_yesno(is_foo and is_garply_bar)
    ```

### Example 6
Q: Write a python code to solve the following coding problem that obeys the constraints. The output code needs to {FEW_SHOT_QUESTION_GUIDE}. Please wrap your code answer using ```: 
Query: Grasp the carrot for me
Plan:
Step 1: Find the carrot in the image.
Step 2: Detect the grasp pose for the first detected carrot.
A: ```
def execute_command(image):
    image_patch = ImagePatch(image)
    carrot_patches = image_patch.find("carrot")
    grasp_pose = image_patch.grasp_detection(carrot_patches[0])
    return grasp_pose
    ```

### Example 7
Q: Write a python code to solve the following coding problem that obeys the constraints. The output code needs to {FEW_SHOT_QUESTION_GUIDE}. Please wrap your code answer using ```: 
Query: Give me the blue bottle
Plan:
Step 1: Find all patches containing bottles in the image.
Step 2: Iterate through each detected bottle patch.
Step 3: Verify if the bottle is blue.
Step 4: Perform grasp pose detection for the blue bottle.
Step 5: Return the grasp pose for the blue bottle.
Step 6: Handle the case where no blue bottles are found. Return None.
A: ```
def execute_command(image):
    image_patch = ImagePatch(image)
    bottle_patches = image_patch.find("bottle")
    for bottle_patch in bottle_patches:
        if bottle_patch.verify_property("bottle", "blue"):
            grasp_pose = image_patch.grasp_detection(bottle_patch)
            return grasp_pose
    return None
    ```

### Example 8
Q: Write a python code to solve the following coding problem that obeys the constraints. The output code needs to {FEW_SHOT_QUESTION_GUIDE}. Please wrap your code answer using ```: 
Qeury: Give me the second chocolate bar from the left
Plan:
Step 1: Find all patches containing chocolate bars in the image.
    -Search for all regions or patches in the image that contain the object "chocolate bar."
Step 2: Sort the chocolate bar patches based on their horizontal position.
    -Sort the detected chocolate bar patches by their horizontal center to arrange them from left to right.
Step 3: Return the grasp pose for the second chocolate bar from the left.
    -After sorting, select the second chocolate bar patch and compute the grasp pose for it.
A: ```
def execute_command(image):
    image_patch = ImagePatch(image)
    bar_patches = image_patch.find("chocolate bar")
    bar_patches.sort(key=lambda x: x.horizontal_center)
    bar_patch = bar_patches[1]
    grasp_pose = image_patch.grasp_detection(bar_patch)
    return grasp_pose
    ```

### Example 9
Q: Write a python code to solve the following coding problem that obeys the constraints. The output code needs to {FEW_SHOT_QUESTION_GUIDE}. Please wrap your code answer using ```: 
Query: Give me the object at the highest position in the image.
Plan:
Step 1: Find all objects in the image.
    -Search for all objects in the image by detecting patches that correspond to the term "object."
Step 2: Sort the objects based on their vertical position.
    -Sort the detected object patches based on their vertical center, which helps in determining the relative position of the objects.
Step 3: Select the object with the highest vertical center.
    -Retrieve the object with the highest vertical center. This is done by selecting the last object in the sorted list. Higher vertical_center means closer to the top of the image.
Step 4: Perform grasp pose detection for the selected object.
    -Once the object is identified, apply grasp pose detection to the selected object to determine the appropriate grasping position.
Step 5: Return the grasp pose for the selected object.
    -Return the calculated grasp pose for the selected object to complete the task.
A: ```
def execute_command(image):
    image_patch = ImagePatch(image)
    object_patches = image_patch.find("object")
    object_patches.sort(key=lambda x: x.vertical_center)
    object_patch = object_patches[-1]
    grasp_pose = image_patch.grasp_detection(object_patch)
    return grasp_pose
    ```

### Example 10
Q: Write a python code to solve the following coding problem that obeys the constraints. The output code needs to {FEW_SHOT_QUESTION_GUIDE}. Please wrap your code answer using ```: 
Query: Give me the knife. Avoid it to harm me. You should grasp the blade, so I could grasp the handle of the knife. It's safe for human.
Plan: 
A: ```
def execute_command(image):
    image_patch = ImagePatch(image)
    knife_patches = image_patch.find("knife")
    knife_patch = building_patches[0]
    knife_blade_patch = knife_patch.find_part("knife", "blade")
    grasp_pose = image_patch.grasp_detection(knife_blade_patch)
    return grasp_pose
    ```

### Example 11
Q: Write a python code to solve the following coding problem that obeys the constraints. The output code needs to {FEW_SHOT_QUESTION_GUIDE}. Please wrap your code answer using ```: 
Please hand me the coffee cup. Since the coffee cup is very hot, you should grasp it by the body, allowing me to safely grasp the handle without risking a burn.
Plan: 
Step 1: Detect all cups in the image.
    -Use the find method of ImagePatch to locate all objects identified as "cup" in the image. This will return a list of patches corresponding to detected cups.
Step 2: Identify the body of the first detected cup.
    -Use the find_part method on the first detected cup patch to locate the "body" part of the cup. This ensures the assistant can grasp the cup by its body, leaving the handle accessible.
Step 3: Perform grasp pose detection for the cup body.
    -Use the grasp_detection method of ImagePatch to calculate the grasp pose for the detected cup body.
Step 4: Return the grasp pose for the cup body.
    -Output the calculated grasp pose as the result, ensuring safe handling of the hot coffee cup.
A: ```
def execute_command(image):
    image_patch = ImagePatch(image)
    cup_patches = image_patch.find("cup")
    cup_body_patch = cup_patches[0].find_part("cup", "body")
    grasp_pose = image_patch.grasp_detection(cup_body_patch)
    return grasp_pose
    ```

### Example 12
Q: Write a python code to solve the following coding problem that obeys the constraints. The output code needs to {FEW_SHOT_QUESTION_GUIDE}. Please wrap your code answer using ```: 
Query: Grasp the cup to pour. In order to pour, you should grasp the handle of the cup.
Plan:
Step 1: Detect all cups in the image.**  
    - Use the `find` method of `ImagePatch` to locate all objects identified as "cup" in the image. This will return a list of patches corresponding to detected cups.
Step 2: Identify the handle of the first detected cup.**  
    - Use the `find_part` method on the first detected cup patch to locate the "handle" part of the cup. Grasping the handle is necessary for safely pouring the contents of the cup.
Step 3: Perform grasp pose detection for the cup handle.**  
    - Use the `grasp_detection` method of `ImagePatch` to calculate the grasp pose for the detected cup handle.
Step 4: Return the grasp pose for the cup handle.**  
    - Output the calculated grasp pose as the result, allowing the assistant to safely grasp the handle for pouring.
A: ```
def execute_command(image):
    image_patch = ImagePatch(image)
    cup_patches = image_patch.find("cup")
    cup_handle_patch = cup_patches[0].find_part("cup", "handle")
    grasp_pose = image_patch.grasp_detection(cup_handle_patch)
    return grasp_pose
    ```
'''

FEW_SHOT_QUESTION_GUIDE='''use Python and the DRONE class to execute and achieve the task in the coding problem'''
llama_context = __context.replace("{FEW_SHOT_QUESTION_GUIDE}", FEW_SHOT_QUESTION_GUIDE)