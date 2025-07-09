from __future__ import annotations
from matplotlib import pyplot as plt
import base64
import os
import numpy as np
from openai import OpenAI
from openai.types import CreateEmbeddingResponse
import requests
import torch
import torchvision
from PIL import Image
import cv2
from typing import List, Tuple
from scipy import spatial
from torchvision import transforms
from torchvision.ops import box_convert
from transformers import (Blip2ForConditionalGeneration, Blip2Processor, 
                          DPTImageProcessor, DPTForDepthEstimation, 
                          AutoModelForMaskGeneration, AutoProcessor, pipeline,
                          Owlv2Processor, Owlv2ForObjectDetection)
from grasp.unit_grasp_pose_generation import detect_grasp, load_grasp_model, get_best_grasp
from vlpart.vlpart import build_vlpart
import detectron2.data.transforms as T

BASE_PATH = os.path.dirname(os.path.realpath(__file__))
query_image_path = os.path.join(BASE_PATH, "imgs/query_image.png")
device = "cuda:0" if torch.cuda.is_available() else "cpu"

## GroundingDINO
BOX_TRESHOLD = 0.35
TEXT_TRESHOLD = 0.25
THRESHOLD = 0.3
detector_id = 'IDEA-Research/grounding-dino-tiny'
# detector_id = 'IDEA-Research/grounding-dino-base'
object_detector = pipeline(model=detector_id, task="zero-shot-object-detection", device=device)

## OWL-V2 model
owl_processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
owl_model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble")


## SAM model
segmenter_id = "facebook/sam-vit-base"
segmentator = AutoModelForMaskGeneration.from_pretrained(segmenter_id).to(device)
segmentor_processor = AutoProcessor.from_pretrained(segmenter_id)

## VLpart model
vlpart_checkpoint = 'weights/swinbase_part_0a0000.pth'
# vlpart_checkpoint = 'weights/swinbase_cascade_lvis_paco_pascalpart_partimagenet.pth'
vlpart = build_vlpart(checkpoint=vlpart_checkpoint)
vlpart.to(device=device)
THRESHOLD_VLPART = 0.4

## BLIP2 model
blip_processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
blip_model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-flan-t5-xl",torch_dtype=torch.bfloat16
).to(device)

## OpenAI API
api_file = os.path.join(BASE_PATH, 'api.key')
with open(api_file) as f:
    api_key = f.readline().splitlines()
OPENAI_CLIENT = OpenAI(api_key=api_key[0]) # , base_url="https://api.deepseek.com"
EMBEDDING_MODEL = "text-embedding-ada-002"

## MiDAS model
depth_id = "Intel/dpt-hybrid-midas"
depth_processor = DPTImageProcessor.from_pretrained(depth_id)
depth_model = DPTForDepthEstimation.from_pretrained(depth_id, low_cpu_mem_usage=True)

### grasp model
grasp_model = load_grasp_model(device = device, ragt_weights_path='weights/RAGT-3-3.pth')

def convert_bbox(image_source: torch.Tensor, boxes: torch.Tensor) -> np.ndarray:
    _,h, w = image_source.shape
    boxes = boxes * torch.Tensor([w, h, w, h])
    xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
    return xyxy

def img_unormalize(img):
    mean = torch.tensor([0.485, 0.456, 0.406]).unsqueeze(1).unsqueeze(1)
    var = torch.tensor([0.229, 0.224, 0.225]).unsqueeze(1).unsqueeze(1)
    img = img*var + mean
    return torch.tensor(img*255, dtype=torch.uint8)

def resize_rectangle(original_size, resized_size, rect):
    original_width, original_height = original_size
    resized_width, resized_height = resized_size

    scale_x = resized_width / original_width
    scale_y = resized_height / original_height

    quality, x, y, w, h, angle = map(float, rect)

    # Scale the rectangle's center coordinates and dimensions
    new_x = x * scale_x
    new_y = y * scale_y
    new_w = w * scale_x*0.6
    new_h = h * scale_y*0.5

    return [quality, new_x, new_y, new_w, new_h, angle]

def mask_to_polygon(mask: np.ndarray) -> List[List[int]]:
    # Find contours in the binary mask
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the contour with the largest area
    largest_contour = max(contours, key=cv2.contourArea)

    # Extract the vertices of the contour
    polygon = largest_contour.reshape(-1, 2).tolist()

    return polygon

def polygon_to_mask(polygon: List[Tuple[int, int]], image_shape: Tuple[int, int]) -> np.ndarray:
    """
    Convert a polygon to a segmentation mask.

    Args:
    - polygon (list): List of (x, y) coordinates representing the vertices of the polygon.
    - image_shape (tuple): Shape of the image (height, width) for the mask.

    Returns:
    - np.ndarray: Segmentation mask with the polygon filled.
    """
    # Create an empty mask
    mask = np.zeros(image_shape, dtype=np.uint8)

    # Convert polygon to an array of points
    pts = np.array(polygon, dtype=np.int32)

    # Fill the polygon with white color (255)
    cv2.fillPoly(mask, [pts], color=(255,))

    return mask

def refine_masks(masks: torch.BoolTensor, polygon_refinement: bool = False) -> List[np.ndarray]:
    masks = masks.cpu().float()
    masks = masks.permute(0, 2, 3, 1)
    masks = masks.mean(axis=-1)
    masks = (masks > 0).int()
    masks = masks.numpy().astype(np.uint8)
    masks = list(masks)

    if polygon_refinement:
        for idx, mask in enumerate(masks):
            shape = mask.shape
            polygon = mask_to_polygon(mask)
            mask = polygon_to_mask(polygon, shape)
            masks[idx] = mask

    return masks

relatedness_fn = lambda x, y: 1 - spatial.distance.cosine(x, y)

class ImagePatch:
    """A Python class containing a crop of an image centered around a particular object, as well as relevant
    information.
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

    Methods
    -------
    find(object_name: str)->List[ImagePatch]
        Returns a list of new ImagePatch objects containing crops of the image centered around any objects found in the
        image matching the object_name.
    exists(object_name: str)->bool
        Returns True if the object specified by object_name is found in the image, and False otherwise.
    verify_property(property: str)->bool
        Returns True if the property is met, and False otherwise.
    best_text_match(option_list: List[str], prefix: str)->str
        Returns the string that best matches the image.
    simple_query(question: str=None)->str
        Returns the answer to a basic question asked about the image. If no question is provided, returns the answer
        to "What is this?".
    compute_depth()->float
        Returns the median depth of the image crop.
    crop(left: int, lower: int, right: int, upper: int)->ImagePatch
        Returns a new ImagePatch object containing a crop of the image at the given coordinates.
    overlaps_with(left: int, lower: int, right: int, upper: int)->bool
        Returns True if a crop with the given coordinates overlaps with this one, else False.
    find_part(object_name: str, part_name: str)->ImagePatch
        Returns a ImagePatch object of the part of the object.
    grasp_detection(mask: np.ndarray)->List[float]
        Returns a best grasp pose with given a mask of an object in the image.
    """

    def __init__(self, image: Image.Image | torch.Tensor | np.ndarray, left: int | None = None, lower: int | None = None,
                 right: int | None = None, upper: int | None = None, parent_left=0, parent_lower=0, queues=None,
                 parent_img_patch=None, mask = None):
        """Initializes an ImagePatch object by cropping the image at the given coordinates and stores the coordinates as
        attributes. If no coordinates are provided, the image is left unmodified, and the coordinates are set to the
        dimensions of the image.

        Parameters
        -------
        image : array_like
            An array-like of the original image.
        left : int
            An int describing the position of the left border of the crop's bounding box in the original image.
        lower : int
            An int describing the position of the bottom border of the crop's bounding box in the original image.
        right : int
            An int describing the position of the right border of the crop's bounding box in the original image.
        upper : int
            An int describing the position of the top border of the crop's bounding box in the original image.

        """

        if isinstance(image, Image.Image):
            image = transforms.ToTensor()(image)
        elif isinstance(image, np.ndarray):
            image = torch.tensor(image).permute(2, 0, 1)
        elif isinstance(image, torch.Tensor) and image.dtype == torch.uint8:
            image = image / 255


        if left is None and right is None and upper is None and lower is None:
            self.cropped_image = image
            self.left = 0
            self.lower = 0
            self.right = image.shape[2]  # width
            self.upper = image.shape[1]  # height
            if mask == None:
                self.mask = torch.ones(self.cropped_image.shape[1],self.cropped_image.shape[2])
            else:
                self.mask = mask
        else:
            self.cropped_image = image[:, image.shape[1]-upper:image.shape[1]-lower, left:right]
            self.left = left + parent_left
            self.upper = upper + parent_lower
            self.right = right + parent_left
            self.lower = lower + parent_lower
            # if mask.any() == None:
            #     self.mask = torch.ones(self.cropped_image.shape[1],self.cropped_image.shape[2])
            # else:
            #     self.mask = torch.from_numpy(mask[image.shape[1]-upper:image.shape[1]-lower, left:right])

        self.height = self.cropped_image.shape[1]
        self.width = self.cropped_image.shape[2]

        # self.cache = {}
        self.queues = (None, None) if queues is None else queues

        self.parent_img_patch = parent_img_patch
        self.mask = mask

        self.horizontal_center = (self.left + self.right) / 2
        self.vertical_center = (self.lower + self.upper) / 2

        if self.cropped_image.shape[1] == 0 or self.cropped_image.shape[2] == 0:
            raise Exception("ImagePatch has no area")

        # if self.height<700 or self.width<700:
        #     scale = max(700/self.width,700/self.height)
        #     self.height = int(self.height*scale)
        #     self.width = int(self.width*scale)
        #     self.cropped_image = torchvision.transforms.Resize((self.height,self.width))(self.cropped_image)
        #     self.mask = torchvision.transforms.Resize((self.height,self.width))(self.mask.unsqueeze(0)).squeeze()

        # draw bounding box and fill color
        # self.PIL_img = img_unormalize(self.cropped_image)
        # transform this image to PIL image
        self.PIL_img = torchvision.transforms.ToPILImage()(self.cropped_image)
        self.original_img = torchvision.transforms.ToPILImage()(image)
        self.original_width = image.shape[2]
        self.original_height = image.shape[1]

        plt.imshow(self.PIL_img)
        plt.axis('off')


    def find(self, object_name: str) -> list[ImagePatch]:
        """Returns a list of ImagePatch objects matching object_name contained in the crop if any are found.
        Otherwise, returns an empty list.
        Parameters
        ----------
        object_name : str
            the name of the object to be found

        Returns
        -------
        List[ImagePatch]
            a list of ImagePatch objects matching object_name contained in the crop
        """
        labels = [label if label.endswith(".") else label+"." for label in [object_name]]
        image = self.PIL_img
        detection_results = object_detector(image, candidate_labels=labels, threshold=THRESHOLD)
        boxes = [[result["box"]['xmin'], 
                  result["box"]['ymin'], 
                  result["box"]['xmax'], 
                  result["box"]['ymax']] 
                for result in detection_results]
        if len(boxes) == 0:
            return [None]
        inputs = segmentor_processor(images=image, input_boxes=[boxes], return_tensors="pt").to(device)
        outputs = segmentator(**inputs)
        masks = segmentor_processor.post_process_masks(
            masks=outputs.pred_masks,
            original_sizes=inputs.original_sizes,
            reshaped_input_sizes=inputs.reshaped_input_sizes
        )[0]
        masks = refine_masks(masks, polygon_refinement=True)
        obj_list = []
        for mask, box in zip(masks, boxes):
            # padding 10% of the width and height of the image patch. We find it better to do it this way for VLpart.
            scale = 0.1
            max_d = max(int(box[2]) - int(box[0]), int(box[3]) - int(box[1]))
            left = max(0, int(box[0]) - scale*max_d)
            right = min(self.width, int(box[2]) + scale*max_d)
            lower = self.height - min(self.height, int(box[3]) + scale*max_d)
            upper = self.height - max(0, int(box[1]) - scale*max_d)
            obj = self.crop(left, lower, right, upper, mask=mask)
            obj_list.append(obj)
        return obj_list

    def exists(self, object_name: str) -> bool:
        """Returns True if the object specified by object_name is found in the image, and False otherwise.
        Parameters
        -------
        object_name : str
            A string describing the name of the object to be found in the image.
        """
        prompt = f"Question: Do you see a {object_name} in the image? Answer (Y/N):"

        self.PIL_img.save(query_image_path,"PNG")
        with open(query_image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')

        response = OPENAI_CLIENT.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt,
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}",
                                        "detail": "low",
                            },
                        },
                    ],
                }
            ],
        )

        response_message = response.choices[0].message.content

        if response_message == 'Y':
            out= True
        else:
            out= False
        return out

    def verify_property(self, object_name: str, attribute: str) -> bool:
        """Returns True if the object possesses the property, and False otherwise.
        Differs from 'exists' in that it presupposes the existence of the object specified by object_name, instead
        checking whether the object possesses the property.
        Parameters
        -------
        object_name : str
            A string describing the name of the object to be found in the image.
        attribute : str
            A string describing the property to be checked.
        """
        
        prompt = f"Is the {object_name} {attribute}? Answer yes or no?"
        ## BLIP
        # inputs = blip_processor(images=self.PIL_img, text=prompt, return_tensors="pt").to(device="cuda", dtype=torch.bfloat16)
        # generated_ids = blip_model.generate(**inputs)
        # generated_text = blip_processor.batch_decode(generated_ids[0], skip_special_tokens=True)

        ## GPT-4o-mini
        self.PIL_img.save(query_image_path,"PNG")
        with open(query_image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        response = OPENAI_CLIENT.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt,
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}",
                                        "detail": "low",
                            },
                        },
                    ],
                }
            ],
        )
        answer = response.choices[0].message.content
        if 'yes' in answer.lower():
            out= True
        else:
            out= False
        return out

    def simple_query(self, question: str):
        """Returns the answer to a basic question asked about the image. If no question is provided, returns the answer
        to "What is this?". The questions are about basic perception, and are not meant to be used for complex reasoning
        or external knowledge.
        Parameters
        -------
        question : str
            A string describing the question to be asked.
        """
        prompt = question
        ### BLIP2
#         inputs = processor(images=self.PIL_img, text=prompt, return_tensors="pt").to(device="cuda", dtype=torch.bfloat16)
#         generated_ids = model_blip.generate(**inputs)
#         generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
#         return generated_text

        ### GPT-4o-mini
        self.PIL_img.save(query_image_path,"PNG")
        with open(query_image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')

        headers = {
          "Content-Type": "application/json",
          "Authorization": f"Bearer {OPENAI_CLIENT.api_key}"
        }

        payload = {
          "model": "gpt-4o-mini",
          "messages": [
            {
              "role": "user",
              "content": [
                {
                  "type": "text",
                  "text": prompt
                },
                {
                  "type": "image_url",
                  "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}",
                    "detail": "low"
                  }
                }
              ]
            }
          ],
          "max_tokens": 300
        }

        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        response_message = response.json()["choices"][0]["message"]["content"]
        return response_message

    def best_image_match(self, list_patches: list[ImagePatch], content: list[str], return_index: bool = False) -> \
            ImagePatch | int | None:
        """Returns the patch most likely to contain the content.
        Parameters
        ----------
        list_patches : List[ImagePatch]
        content : List[str]
            the object of interest<
        return_index : bool
            if True, returns the index of the patch most likely to contain the object

        Returns
        -------
        int
            Patch most likely to contain the object
        """

        if len(list_patches) == 0:
            return None

        patch_embeddings: list[CreateEmbeddingResponse] = []
        for patch in list_patches:
            inputs = blip_processor(images=patch.PIL_img, return_tensors="pt").to(device="cuda", dtype=torch.bfloat16)
            generated_ids = blip_model.generate(**inputs)
            generated_text = blip_processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            ## convert generated_text to embedding
            response = OPENAI_CLIENT.embeddings.create(model=EMBEDDING_MODEL, input=generated_text)
            patch_embeddings.append(response)

        scores = torch.zeros(len(patch_embeddings))
        for cont in content:
            query_embedding = OPENAI_CLIENT.embeddings.create(model=EMBEDDING_MODEL, input=cont)
            relatedness = [relatedness_fn(query_embedding.data[0].embedding, embed.data[0].embedding) for embed in patch_embeddings]
            scores += torch.tensor(relatedness)
        scores = scores / len(content)

        scores = scores.argmax().item()  # Argmax over all image patches

        if return_index:
            return scores
        return list_patches[scores]

    def overlaps_with(self, left, lower, right, upper):
        """Returns True if a crop with the given coordinates overlaps with this one,
        else False.
        Parameters
        ----------
        left : int
            the left border of the crop to be checked
        lower : int
            the lower border of the crop to be checked
        right : int
            the right border of the crop to be checked
        upper : int
            the upper border of the crop to be checked

        Returns
        -------
        bool
            True if a crop with the given coordinates overlaps with this one, else False
        """
        return self.left <= right and self.right >= left and self.lower <= upper and self.upper >= lower

    def crop(self, left: int, lower: int, right: int, upper: int, mask) -> ImagePatch:
        """Returns a new ImagePatch containing a crop of the original image at the given coordinates.
        Parameters
        ----------
        left : int
            the position of the left border of the crop's bounding box in the original image
        lower : int
            the position of the bottom border of the crop's bounding box in the original image
        right : int
            the position of the right border of the crop's bounding box in the original image
        upper : int
            the position of the top border of the crop's bounding box in the original image

        Returns
        -------
        ImagePatch
            a new ImagePatch containing a crop of the original image at the given coordinates
        """
        # make all inputs ints
        left = int(left)
        lower = int(lower)
        right = int(right)
        upper = int(upper)
        return ImagePatch(self.cropped_image, left, lower, right, upper, self.left, self.lower, queues=self.queues,
                          parent_img_patch=self, mask = mask)


    def llm_query(self, question, context=None, long_answer=True, queues=None):
        """Answers a text question using GPT-4o-mini. The input question is always a formatted string with a variable in it.

        Parameters
        ----------
        query: str
            the text question to ask. Must not contain any reference to 'the image' or 'the photo', etc.
        """
        # query = question.format(object_name)
        # self.PIL_img.save(img_path,"PNG")
        self.original_img.save(query_image_path,"PNG")
        with open(query_image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')

        response = OPENAI_CLIENT.chat.completions.create(
          model="gpt-4o",
          messages=[
            {
              "role": "user",
              "content": [
                {
                  "type": "text",
                  "text": question,
                },
                {
                  "type": "image_url",
                  "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}",
                    "detail": "high"
                  }
                }
              ]
            }
          ],
          max_tokens=300,
        )

        response_message = response.choices[0].message.content
        return response_message

    def bool_to_yesno(self, bool_answer: bool) -> str:
        """Returns a yes/no answer to a question based on the boolean value of bool_answer.
        Parameters
        ----------
        bool_answer : bool
            a boolean value

        Returns
        -------
        str
            a yes/no answer to a question based on the boolean value of bool_answer
        """
        return "yes" if bool_answer else "no"

    def compute_depth(self):
        """Returns the median depth of the image crop
        Parameters
        ----------
        Returns
        -------
        float
            the median depth of the image crop
        """
        image = self.PIL_img
        inputs = depth_processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = depth_model(**inputs)
            predicted_depth = outputs.predicted_depth

        # interpolate to original size
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=image.size[::-1],
            mode="bicubic",
            align_corners=False,
        )

        output = prediction.squeeze().cpu().numpy()
        return int(np.median(output))  # Ideally some kind of mode, but median is good enough for now

    def find_part(self, object_name: str, part_name: str) -> ImagePatch:
        """Returns a ImagePatch object of the part of the object
        Parameters
        ----------
        object_name: 
            name of the object
        part_name: 
            name of the part of the object
            
        Returns
        -------
        ImagePatch
            a ImagePatch object of the part of the object (in size of original image)
        """
        image_np = np.array(self.PIL_img)
        preprocess = T.ResizeShortestEdge([800, 800], 1333)
        height, width = image_np.shape[:2]
        image = preprocess.get_transform(image_np).apply_image(image_np)
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
        inputs = {"image": image, "height": height, "width": width}
        
        text_prompt = f'{object_name} {part_name}'
        with torch.no_grad():
            predictions = vlpart.inference([inputs], text_prompt=text_prompt)[0]
            
        boxes, masks = None, None
        filter_scores, filter_boxes, filter_classes = [], [], []

        if "instances" in predictions:
            instances = predictions['instances'].to('cpu')
            boxes = instances.pred_boxes.tensor if instances.has("pred_boxes") else None
            scores = instances.scores if instances.has("scores") else None
            classes = instances.pred_classes.tolist() if instances.has("pred_classes") else None

            num_obj = len(scores)
            for obj_ind in range(num_obj):
                category_score = scores[obj_ind]
                if category_score < THRESHOLD_VLPART:
                    continue
                filter_scores.append(category_score)
                filter_boxes.append(boxes[obj_ind])
                filter_classes.append(classes[obj_ind])
        if len(filter_boxes) == 0:
            return self.crop(0, 0, self.width, self.height, mask=self.mask) #return the object ifself
        inputs = segmentor_processor(images=image_np, input_boxes=[[filter_boxes]], return_tensors="pt").to(device)

        outputs = segmentator(**inputs)
        masks = segmentor_processor.post_process_masks(
            masks=outputs.pred_masks,
            original_sizes=inputs.original_sizes,
            reshaped_input_sizes=inputs.reshaped_input_sizes
        )[0]

        masks = refine_masks(masks, polygon_refinement=True)[0] # numpy unit8
        # return to oringinal size
        mask_original = np.zeros_like(self.mask)
        mask_original[self.original_height-self.upper:self.original_height-self.lower, self.left:self.right] = masks
        left, lower, right, upper = map(int, filter_boxes[0])
        obj = self.crop(left, self.height-upper, right, self.height-lower, mask=mask_original)
        return obj
        
    def grasp_detection(self, object_patch):
        """Return a best grasp pose with given a mask of an object in the image
        Parameters
        ----------
        mask: 
            a mask of an object or part of an object in the image

        Returns
        -------
        List[float]
            a grasp pose is a rectangle represented by [quality, x, y, w, h, angle], 
            where x,y is the position of center of the rectangle, w and h is the width and height of the rectangle
            and angle is the angle of the rectangle
        """
        if object_patch is None:
            return None
        if isinstance(object_patch, list):
            return None
        image = np.array(self.PIL_img)
        mask = object_patch.mask
        image_resized = cv2.resize(image, (416, 416))
        mask_resized = cv2.resize(mask, (416, 416))
        grasp_pose = detect_grasp(grasp_model, image_resized, mask_resized, device).detach().cpu().numpy()
        grasp_pose = resize_rectangle((416, 416), (image.shape[1], image.shape[0]), grasp_pose)
        return grasp_pose

    # def get_original_boxes(self):
    #     [[dog_patch.left,image_patch.height-dog_patch.upper,dog_patch.right,image_patch.height-dog_patch.lower]]


    # def get_mask(self, object_name: str, box: list):
    #     """Returns a mask of the object in question
    #     Parameters
    #     -------
    #     object name : str
    #         A string describing the name of the object to be masked in the image.
    #     object name : list
    #         Optional list of bounding box values of object.
    #     >>> # Generate the mask of the kid raising their hand
    #     >>> def execute_command(image) -> str:
    #     >>>     image_patch = ImagePatch(image)
    #     >>>     kid_patches = image_patch.find("kid")
    #     >>>     for kid_patch in kid_patches:
    #     >>>         if kid_patch.verify_property("kid", "raised hand"):
    #     >>>             return image_patch.get_mask("kid",[[kid_patch.left,image_patch.height-kid_patch.upper,kid_patch.right,image_patch.height-kid_patch.lower]])
    #     >>>     return None
    #     """
    #     return masks