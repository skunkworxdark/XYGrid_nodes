# Copyright (c) 2023 skunkworxdark (https://github.com/skunkworxdark)

from typing import Literal, Optional, Union

from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel, Field
from itertools import product
import json
import re

from ..models.image import ImageCategory, ImageField, ResourceOrigin, ColorField
from .baseinvocation import BaseInvocation, BaseInvocationOutput, InvocationContext, InvocationConfig
from .image import PILInvocationConfig, PIL_RESAMPLING_MODES ,PIL_RESAMPLING_MAP


class FloatsToStringsOutput(BaseInvocationOutput):
    """FloatsToStringsOutput"""
    type: Literal["float_to_string_output"] = "float_to_string_output"
    float_string: list[str] = Field(default=[], description="collection of strings")

    class Config:
        schema_extra = {"required": ["type", "float_string"]}
    
class FloastToStringsInvocation(BaseInvocation):
    """FloatsToStrings converts a float or collections of floats to a collection of strings"""
    type: Literal["floats_to_strings"] = "floats_to_strings"
    floats: Union[float, list[float], None] = Field(default=None, description="float or collection of floats")

    class Config(InvocationConfig):
        schema_extra = {"ui": {"title": "Floats To Strings","type_hints": {"floats": "float"}}}

    def invoke(self, context: InvocationContext) -> FloatsToStringsOutput:
        """Invoke with provided services and return outputs."""
        if self.floats is None:
            raise Exception("No collection of floats provided")
        if isinstance(self.floats, list):
            return FloatsToStringsOutput(float_string=[str(x) for x in self.floats])
        else:
            return FloatsToStringsOutput(float_string=[str(self.floats)])


class StringToFloatOutput(BaseInvocationOutput):
    """StringToFloatOutput"""
    type: Literal["strings_to_floats_output"] = "strings_to_floats_output"
    floats: float = Field(default=1.0, description="float")

    class Config:
        schema_extra = {"required": ["type", "floats"]}
    
class StringToFloatInvocation(BaseInvocation):
    """StringToFloat converts a string to a float"""
    type: Literal["string_to_float"] = "string_to_float"
    float_string: str = Field(default='', description="string")

    class Config(InvocationConfig):
        schema_extra = {"ui": {"title": "String To Float"}}

    def invoke(self, context: InvocationContext) -> StringToFloatOutput:
        """Invoke with provided services and return outputs."""
        return StringToFloatOutput(floats=float(self.float_string))


class IntsToStringsOutput(BaseInvocationOutput):
    """IntsToStringsOutput"""
    type: Literal["ints_to_strings_output"] = "ints_to_strings_output"
    int_string: list[str] = Field(default=[], description="collection of strings")

    class Config:
        schema_extra = {"required": ["type", "int_string"]}
    
class IntsToStringsInvocation(BaseInvocation):
    """IntsToStrings converts an int or collection of ints to a collection of strings"""
    type: Literal["ints_to_strings"] = "ints_to_strings"
    ints: Union[int, list[int], None] = Field(default=None, description="int or collection of ints")

    class Config(InvocationConfig):
        schema_extra = {"ui": {"title": "Ints To Strings", "type_hints": {"ints": "integer"}}}

    def invoke(self, context: InvocationContext) -> IntsToStringsOutput:
        """Invoke with provided services and return outputs."""
        if self.ints is None:
            raise Exception("No collection of ints provided")
        if isinstance(self.ints, list):
            return IntsToStringsOutput(int_string=[str(x) for x in self.ints])
        else:
            return IntsToStringsOutput(int_string=[str(self.ints)])


class StringToIntOutput(BaseInvocationOutput):
    """StringToIntOutput"""
    type: Literal["string_to_int_output"] = "string_to_int_output"
    ints: int = Field(default=1, description="int")

    class Config:
        schema_extra = {"required": ["type", "ints"]}
    
class StringToIntInvocation(BaseInvocation):
    """StringToInt converts a string to an int"""
    type: Literal["string_to_int"] = "string_to_int"
    int_string: str = Field(default='', description="string")

    class Config(InvocationConfig):
        schema_extra = {"ui": {"title": "String To Int"}}

    def invoke(self, context: InvocationContext) -> StringToIntOutput:
        """Invoke with provided services and return outputs."""
        return StringToIntOutput(ints=int(self.int_string))


class XYCollectOutput(BaseInvocationOutput):
    """XYCollectOutput a collection that contains every combination of the input collections"""
    type: Literal["xy_collect_output"] = "xy_collect_output"
    xy_collection: list[list[str]] = Field(description="The x y product collection")

    class Config:
        schema_extra = {"required": ["type", "xy_collection"]}

class XYCollectInvocation(BaseInvocation):
    """XYCollect takes two string collections and outputs a collection that every combination of the inputs"""
    type: Literal["xy_collect"] = "xy_collect"
    x_collection: list[str] = Field(default=[], description="The X collection")
    y_collection: list[str] = Field(default=[], description="The Y collection")

    class Config(InvocationConfig):
        schema_extra = {"ui": {"title": "XY Collect"}}

    def invoke(self, context: InvocationContext) -> XYCollectOutput:
        """Invoke with provided services and return outputs."""
        return XYCollectOutput(xy_collection=list(product(self.x_collection, self.y_collection)))

# Was going to try have a collection node that could remove the need for a To String Node.
# Not working at the moment.
#
# class XYCollect2Invocation(BaseInvocation):
#     """class for XYCollectionExpand a collection that contains every combination of the input collections"""

#     type: Literal["xy_collect2"] = "xy_collect2"

#     x_ints: Union[int, list[int], None] = Field(default=None, description="The X int collection")
#     x_floats: Union[float, list[float], None] = Field(default=None, description="The X float collection")
#     y_ints: Union[int, list[int], None] = Field(default=None, description="The y int collection")
#     y_floats: Union[float, list[float], None] = Field(default=None, description="The y float collection")

#     class Config(InvocationConfig):
#         schema_extra = {
#             "ui": {
#                 "title": "XY Collect",
#                 "type_hints": {
#                 }
#             },
#         }

#     def invoke(self, context: InvocationContext) -> XYCollectOutput:
#         """Invoke with provided services and return outputs."""
#         return XYCollectOutput(xy_collection=list(product(self.x_collection, self.y_collection)))


class XYExpandOutput(BaseInvocationOutput):
    """XYExpandOutput two strings that are expanded from a collection of strings"""
    type: Literal["xy_expand_output"] = "xy_expand_output"
    x_item: str = Field(description="The X item")
    y_item: str = Field(description="The y item")

    class Config:
        schema_extra = {'required': ['type','x_item','y_item']}  

class XYExpandInvocation(BaseInvocation):
    """XYExpand takes a collection of strings and outputs the first two elements and outputs as individual strings"""
    type: Literal["xy_expand"] = "xy_expand"
    xy_collection: list[str] = Field(default=[], description="The XY collection item")

    class Config(InvocationConfig):
        schema_extra = {"ui": {"title": "XY Expand"}}

    def invoke(self, context: InvocationContext) -> XYExpandOutput:
        """Invoke with provided services and return outputs"""
        return XYExpandOutput(x_item=self.xy_collection[0], y_item=self.xy_collection[1])


class XYImageCollectOutput(BaseInvocationOutput):
    """XYImageCollectOutput string containg an array of xItem, Yitem, Image_name converted to json"""
    type: Literal["xyimage_collect_output"] = "xyimage_collect_output"
    xyimage: str = Field(description="The XY Image ")

    class Config:
        schema_extra = {'required': ['type','xyimage']}

class XYImageCollectInvocation(BaseInvocation):
    """XYImageCollect takes XItem, YItem and an Image and outputs it as an (x_item,y_item,image_name)array converted to json"""
    type: Literal["xyimage_collect"] = "xyimage_collect"
    x_item: str = Field(default='', description="The X item")
    y_item: str = Field(default='', description="The Y item")
    image: ImageField = Field(default=None, description="The image to turn into grids")

    class Config(InvocationConfig):
        schema_extra = {"ui": {"title": "XYImage Collect"}}

    def invoke(self, context: InvocationContext) -> XYImageCollectOutput:
        """Invoke with provided services and return outputs."""
        return XYImageCollectOutput(xyimage = json.dumps([self.y_item, self.x_item , self.image.image_name]))


class XYImageToGridOutput(BaseInvocationOutput):
    """XYImageToGridOutput collection of image grids generated"""
    type: Literal["xyimage_grid_output"] = "xyimage_grid_output"
    collection: list[ImageField] = Field(default=[], description="The output images")

    class Config:
        schema_extra = {"required": ["type", "collection"]}

class XYImagesToGridInvocation(BaseInvocation):#, PILInvocationConfig):
    """Load a collection of xyimage types (json of (x_item,y_item,image_name)array) and create a gridimage of them"""
    type: Literal["xyimage_grid"] = "xyimage_grid"
    xyimages: list[str] = Field(default=[], description="The xyImage Collection")
    space: int = Field(default=1, ge=0, description="The space to be added between images")
    scale_factor: Optional[float] = Field(default=1.0, gt=0, description="The factor by which to scale the images")
    resample_mode:  PIL_RESAMPLING_MODES = Field(default="bicubic", description="The resampling mode")
    background_color: ColorField = Field(
        default=ColorField(r=0, g=0, b=0, a=255),
        description="The color to use as the background",
    )
    label_font_name: str = Field(default="arial.ttf", description="Name of the font to use for labels")
    label_font_size: int = Field(default=35, description="Size of the font to use for labels")
    top_label_height: int = Field(default=50, description="Height of the top label area")
    left_label_width: int = Field(default=100, description="Width of the left label area")
    label_font_color: ColorField = Field(
        default=ColorField(r=255, g=255, b=255, a=255),
        description="The color to use for the label font",
    )

    class Config(InvocationConfig):
        schema_extra = {"ui": {"title": "XYImage To Grid"}}
    
    def is_all_numeric(self,array):
        pattern = r'^-?\d+(\.\d+)?$'
        return all(re.match(pattern, item) for item in array)

    def sort_array(self, arr):
        def sort_key(x):
            key0 = float(x) if self.is_all_numeric(arr) else x[0]
            return (key0)
        return sorted(arr, key=sort_key)
    
    def is_all_numeric2(self,array, i):
        pattern = r'^-?\d+(\.\d+)?$'
        return all(re.match(pattern, item[i]) for item in array)

    def sort_array2(self, arr):
        def sort_key2(x):
            key0 = float(x[0]) if self.is_all_numeric2(arr,0) else x[0]
            key1 = float(x[1]) if self.is_all_numeric2(arr,1) else x[1]
            return (key0, key1)
        return sorted(arr, key=sort_key2)

    def invoke(self, context: InvocationContext) -> XYImageToGridOutput:
        """Convert an image list into a grids of images"""

        top_label_space = self.top_label_height
        left_label_space = self.left_label_width
        text_color = self.label_font_color.tuple()
        font =  ImageFont.truetype(self.label_font_name, self.label_font_size)

        new_array = [json.loads(s) for s in self.xyimages]
        sorted_array = self.sort_array2(new_array)
        images = [context.services.images.get_pil_image(item[2]) for item in sorted_array]
        row_names = self.sort_array(set([item[0] for item in sorted_array]))
        rows = len(row_names)
        column_names = self.sort_array(set([item[1] for item in sorted_array]))
        columns = len(column_names)
        width_max = int(max([image.width for image in images]) * self.scale_factor)
        height_max = int(max([image.height for image in images]) * self.scale_factor)
        background_width = width_max * columns + (self.space * (columns - 1)) + left_label_space
        background_height = height_max * rows + (self.space * (rows - 1)) + top_label_space
        resample_mode = PIL_RESAMPLING_MAP[self.resample_mode]

        column = 0
        row = 0
        x_offset = left_label_space
        y_offset = top_label_space
        background = Image.new('RGBA', (background_width, background_height), self.background_color.tuple())
        grid_images = []

        for image in images:
            if not self.scale_factor == 1.0:
                image = image.resize(
                    (int(image.width * self.scale_factor), int(image.width * self.scale_factor)),
                    resample=resample_mode,
                )

            background.paste(image, (x_offset, y_offset))

            column += 1
            x_offset += width_max + self.space
            if column >= columns:
                column = 0
                x_offset = left_label_space
                y_offset += height_max + self.space
                row += 1
            
            if row >= rows:
                row = 0
                y_offset = top_label_space

                draw = ImageDraw.Draw(background)
                labelx = left_label_space
                labely =  0 
                for label in column_names:
                    text_bbox = font.getbbox(label)
                    text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[0]
                    text_x = labelx + (width_max - text_width) // 2
                    text_y = labely + (top_label_space - text_height) // 2
                    draw.text((text_x, text_y), label, fill=text_color, font=font)
                    labelx += width_max

                labelx = 0
                labely = top_label_space
                for label in row_names:
                    text_bbox = font.getbbox(label)
                    text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[0]
                    text_x = labelx + (left_label_space - text_width) // 2
                    text_y = labely + (height_max - text_height) // 2
                    draw.text((text_x, text_y), label, fill=text_color, font=font)
                    labely += height_max

                image_dto = context.services.images.create(
                    image=background,
                    image_origin=ResourceOrigin.INTERNAL,
                    image_category=ImageCategory.GENERAL,
                    node_id=self.id,
                    session_id=context.graph_execution_state_id,
                    is_intermediate=self.is_intermediate,
                )
                grid_images.append(ImageField(image_name=image_dto.image_name))
                background = Image.new('RGBA', (background_width, background_height), self.background_color.tuple())
        
        if column > 0 or row > 0:
            image_dto = context.services.images.create(
                image=background,
                image_origin=ResourceOrigin.INTERNAL,
                image_category=ImageCategory.GENERAL,
                node_id=self.id,
                session_id=context.graph_execution_state_id,
                is_intermediate=self.is_intermediate,
            )
            grid_images.append(ImageField(image_name=image_dto.image_name))
        


        return XYImageToGridOutput(collection=grid_images)


class ImagesToGridsOutput(BaseInvocationOutput):
    """ImagesToGridsOutput that output nothing"""
    type: Literal["image_grid_output"] = "image_grid_output"
    collection: list[ImageField] = Field(default=[], description="The output images")

    class Config:
        schema_extra = {"required": ["type", "collection"]}

class ImagesToGridsInvocation(BaseInvocation, PILInvocationConfig):
    """Load a collection of images and creat grid images from it and output a collection of genereated grid images"""
    type: Literal["image_grid"] = "image_grid"
    images: list[ImageField] = Field(default=[], description="The image collection to turn into grids")
    columns: int = Field(default=1, ge=1, description="The number of columns in each grid")
    rows: int = Field(default=1, ge=1, description="The nuber of rows to have in each grid")
    space: int = Field(default=1, ge=0, description="The space to be added between images")
    scale_factor: Optional[float] = Field(default=1.0, gt=0, description="The factor by which to scale the images")
    resample_mode:  PIL_RESAMPLING_MODES = Field(default="bicubic", description="The resampling mode")
    background_color: ColorField = Field(
        default=ColorField(r=0, g=0, b=0, a=255),
        description="The color to use as the background",
    )

    class Config(InvocationConfig):
        schema_extra = {"ui": {"title": "Images To Grids", "type_hints": {"images": "image_collection"}}}

    def invoke(self, context: InvocationContext) -> ImagesToGridsOutput:
        """Convert an image list into a grids of images"""
        images = [context.services.images.get_pil_image(image.image_name) for image in self.images]
        width_max = int(max([image.width for image in images]) * self.scale_factor)
        height_max = int(max([image.height for image in images]) * self.scale_factor)
        background_width = width_max * self.columns + (self.space * (self.columns - 1))
        background_height = height_max * self.rows + (self.space * (self.rows - 1))
        resample_mode = PIL_RESAMPLING_MAP[self.resample_mode]

        column = 0
        row = 0
        x_offset = 0
        y_offset = 0
        background = Image.new('RGBA', (background_width, background_height), self.background_color.tuple())
        grid_images = []

        for image in images:
            if not self.scale_factor == 1.0:
                image = image.resize(
                    (int(image.width * self.scale_factor), int(image.width * self.scale_factor)),
                    resample=resample_mode,
                )

            background.paste(image, (x_offset, y_offset))

            column += 1
            x_offset += width_max + self.space
            if column >= self.columns:
                column = 0
                x_offset = 0
                y_offset += height_max + self.space
                row += 1
            
            if row >= self.rows:
                row = 0
                y_offset = 0
                image_dto = context.services.images.create(
                    image=background,
                    image_origin=ResourceOrigin.INTERNAL,
                    image_category=ImageCategory.GENERAL,
                    node_id=self.id,
                    session_id=context.graph_execution_state_id,
                    is_intermediate=self.is_intermediate,
                )
                grid_images.append(ImageField(image_name=image_dto.image_name))
                background = Image.new('RGBA', (background_width, background_height), self.background_color.tuple())
        
        if column > 0 or row > 0:
            image_dto = context.services.images.create(
                image=background,
                image_origin=ResourceOrigin.INTERNAL,
                image_category=ImageCategory.GENERAL,
                node_id=self.id,
                session_id=context.graph_execution_state_id,
                is_intermediate=self.is_intermediate,
            )
            grid_images.append(ImageField(image_name=image_dto.image_name))

        return ImagesToGridsOutput(collection=grid_images)
