CODE = '''

** Role **: You are an expert software programmer. Your task is to convert the given step-by-step instructin plan into executable Python code.
Ensure your code follow the plan. 

** Important instructions **:
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
        An int describing the position of the left border of the crop's bounding box in the original image. Higher values are closer to the left.
    lower : int
        An int describing the position of the bottom border of the crop's bounding box in the original image. Higher values are closer to the bottom.
    right : int
        An int describing the position of the right border of the crop's bounding box in the original image. Higher values are closer to the right.
    upper : int
        An int describing the position of the top border of the crop's bounding box in the original image. Higher values are closer to the top.
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
    simple_query(question: str=None)->str
        Returns the answer to a basic question asked about the image. If no question is provided, returns the answer to "What is this?".
    compute_depth()->float
        Returns the median depth of the image crop.
    crop(left: int, lower: int, right: int, upper: int)->ImagePatch
        Returns a new ImagePatch object containing a crop of the image at the given coordinates.
    find_part(object_name: str, part_name: str)->ImagePatch
        Returns a new ImagePatch object containing crops of the image centered around a part of an object (object_name) in the image that
        matching the part_name 
    llm_query(question: str)->str
        Returns the answer to a question asked about the image using the LLM model. Typical use when the question is complex, ambiguous, or requires external knowledge. 
        Typically ask about the object properties, relationships between them. For example: Ask the color of the Kleenex package in the image.
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
        """
        return find_in_image(self.cropped_image, object_name)

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

    def llm_query(self, question: str) -> str:
        """Returns the answer to a question asked about the image using the LLM model. Typical use when the question is complex, ambiguous, or requires external knowledge.
        Parameters
        -------
        question : str
            A string describing the question to be asked.
        
        Returns
        -------
        str

        Examples
        -------
        >>> # What is the color of the Kleenex package in the image?
        >>> def execute_command(image) -> str:
        >>>     image_patch = ImagePatch(image)
        >>>     return image_patch.llm_query("What is the color of the Kleenex package in the image?")
        """
        return llm_query(question)

Write a function using Python and the ImagePatch class (above) that could be executed to provide an answer to the query. 


### Examples
{example}

Plan at this step: {plan}
** Expected format output begin with **
def execute_command(image):
'''

EXAMPLES_CODER = '''
### Example 1
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

### Example 2
Plan:
Step 1: Find all patches containing bottles in the image.
Step 2: Iterate through each detected bottle patch. 
Step 3: Verify if the bottle is both blue and red.
Step 4: Perform grasp pose detection for the blue bottle.
Step 5: Handle the case where no blue bottles are found. Return None.
A: ```
def execute_command(image):
    image_patch = ImagePatch(image)
    bottle_patches = image_patch.find("bottle")
    for bottle_patch in bottle_patches:
        if bottle_patch.verify_property("bottle", "blue and red"):
                grasp_pose = image_patch.grasp_detection(bottle_patch)
                return grasp_pose
    return None
    ```

### Example 3
Plan:
Step 1: Find all patches containing chocolate bars in the image.
Step 2: Sort the chocolate bar patches based on their horizontal position.
Step 3: The second chocolate bar from the left will be the second element in the sorted list.
Step 4: Return the grasp pose for the second chocolate bar from the left.
A: ```
def execute_command(image):
    image_patch = ImagePatch(image)
    bar_patches = image_patch.find("chocolate bar")
    bar_patches.sort(key=lambda x: x.horizontal_center)
    bar_patch = bar_patches[1]
    grasp_pose = image_patch.grasp_detection(bar_patch)
    return grasp_pose
    ```

### Example 4
Plan:
Step 1: Find all apples in the image.
Step 2: Sort the apples patches based on their vertical position.
Step 3: The apple at highest position is the last item in the list.
Step 4: Return the grasp pose for the highest position apple.
A: ```
def execute_command(image):
    image_patch = ImagePatch(image)
    apple_patches = image_patch.find("apple")
    apple_patches.sort(key=lambda x: x.vertical_center)
    apple_patch = apple_patches[-1]
    grasp_pose = image_patch.grasp_detection(apple_patch)
    return grasp_pose
    ```

### Example 5
Plan: 
Step 1: Detect all knives in the image.
Step 2: To handover to the user safely, locate the "blade" part of the first detected knife.
Step 3: Calculate the grasp pose for the knife blade.
A: ```
def execute_command(image):
    image_patch = ImagePatch(image)
    knife_patches = image_patch.find("knife")
    knife_patch = building_patches[0]
    knife_blade_patch = knife_patch.find_part("knife", "blade")
    grasp_pose = image_patch.grasp_detection(knife_blade_patch)
    return grasp_pose
    ```

### Example 6
Plan: 
Step 1: Question about the Kleenex box in the image, find out its color.
A: ```
def execute_command(image):
    image_patch = ImagePatch(image)
    kleenex_info = image_patch.llm_query("What is the color of the Kleenex package in the image?")
    return kleenex_info
    ```
'''