# 2023 skunkworxdark (https://github.com/skunkworxdark)

from typing import Literal, Optional, Union
from PIL import Image, ImageDraw, ImageFont
from itertools import product

import json
import re

from invokeai.app.models.image import ImageCategory, ResourceOrigin
from invokeai.app.invocations.image import PIL_RESAMPLING_MODES ,PIL_RESAMPLING_MAP
from invokeai.app.invocations.primitives import (
    StringOutput,
    StringCollectionOutput,
    FloatOutput,
    IntegerOutput,
    ColorField,
    ImageField,
    ImageCollectionOutput,
)
from invokeai.app.invocations.baseinvocation import (
    BaseInvocation,
    BaseInvocationOutput,
    InputField,
    Input,
    OutputField,
    InvocationContext,
    UIComponent,
    UIType,
    invocation,
    invocation_output,
)


@invocation("floats_to_strings", title="Floats To String", tags=["float"], category="util")
class FloastToStringsInvocation(BaseInvocation):
    """FloatsToStrings converts a float or collections of floats to a collection of strings"""
    floats: Union[float, list[float], None] = InputField(default=None, description="float or collection of floats", ui_type=UIType.Collection, input=Input.Connection)

    def invoke(self, context: InvocationContext) -> StringCollectionOutput:
        if self.floats is None:
            raise Exception("No collection of floats provided")
        if isinstance(self.floats, list):
            return StringCollectionOutput(collection=[str(x) for x in self.floats])
        else:
            return StringCollectionOutput(collection=[str(self.floats)])


@invocation("string_to_float", title="String To Float", tags=["float"], category="util")
class StringToFloatInvocation(BaseInvocation):
    """StringToFloat converts a string to a float"""
    float_string: str = InputField(default='', description="string containg a float to convert")

    def invoke(self, context: InvocationContext) -> FloatOutput:
        return FloatOutput(value=float(self.float_string))


@invocation("ints_to_strings", title="Ints To String", tags=["int"], category="util")
class IntsToStringsInvocation(BaseInvocation):
    """IntsToStrings converts an int or collection of ints to a collection of strings"""
    ints: Union[int, list[int], None] = InputField(default=None, description="int or collection of ints", ui_type=UIType.Collection, input=Input.Connection)

    def invoke(self, context: InvocationContext) -> StringCollectionOutput:
        if self.ints is None:
            raise Exception("No collection of ints provided")
        if isinstance(self.ints, list):
            return StringCollectionOutput(collection=[str(x) for x in self.ints])
        else:
            return StringCollectionOutput(collection=[str(self.ints)])

   
@invocation("string_to_int", title="String To Int", tags=["int"], category="util")
class StringToIntInvocation(BaseInvocation):
    """StringToInt converts a string to an int"""
    int_string: str = InputField(default='', description="string containing an integer to convert")

    def invoke(self, context: InvocationContext) -> IntegerOutput:
        return IntegerOutput(value=int(self.int_string))


@invocation_output("xy_collect_output")
class XYCollectOutput(BaseInvocationOutput):
    """XYCollectOutput a collection that contains every combination of the input collections"""
    xy_collection: list[list[str]] = OutputField(description="The x y product collection", ui_type=UIType.Collection)


@invocation("xy_collect", title="XY Collect", tags=["xy", "grid", "collect"], category="grid")
class XYCollectInvocation(BaseInvocation):
    """XYCollect takes two string collections and outputs a collection that every combination of the inputs"""
    x_collection: list[str] = InputField(default=[], description="The X collection", ui_type=UIType.Collection, input=Input.Connection)
    y_collection: list[str] = InputField(default=[], description="The Y collection", ui_type=UIType.Collection, input=Input.Connection)

    def invoke(self, context: InvocationContext) -> XYCollectOutput:
        return XYCollectOutput(xy_collection=list(product(self.x_collection, self.y_collection)))


@invocation("xy_csv_to_strings", title="XY CSV to Strings", tags=["xy", "grid", "csv"], category="grid")
class XYCSVToStringsInvocation(BaseInvocation):
    """XYCSVToStrings converts X and Y CSV Strings to a collection that every combination of X and Y"""
    x: str = InputField(default='', description="x string", ui_component=UIComponent.Textarea)
    y: str = InputField(default='', description="y string", ui_component=UIComponent.Textarea)

    def invoke(self, context: InvocationContext) -> XYCollectOutput:
        return XYCollectOutput(xy_collection=list(product(self.x.split(","), self.y.split(","))))


@invocation("csv_to_strings", title="CSV To Strings", tags=["xy", "grid", "csv"], category="grid")
class CSVToStringsInvocation(BaseInvocation):
    """CSVToStrings converts a CSV String to a collection of strings"""
    csv: str = InputField(default='', description="csv string")

    def invoke(self, context: InvocationContext) -> StringCollectionOutput:
        return StringCollectionOutput(collection=self.csv.split(","))


@invocation_output("xy_expand_output")
class XYExpandOutput(BaseInvocationOutput):
    """XYExpandOutput two strings that are expanded from a collection of strings"""
    x_item: str = OutputField(description="The X item")
    y_item: str = OutputField(description="The y item")

@invocation("xy_expand", title="XY Expand", tags=["xy", "grid"], category="grid")
class XYExpandInvocation(BaseInvocation):
    """XYExpand takes a collection of strings and outputs the first two elements and outputs as individual strings"""
    xy_collection: list[str] = InputField(default=[], description="The XY collection item", ui_type=UIType.Collection, input=Input.Connection)

    def invoke(self, context: InvocationContext) -> XYExpandOutput:
        return XYExpandOutput(x_item=self.xy_collection[0], y_item=self.xy_collection[1])


@invocation("xyimage_collect", title="XYImage Collect", tags=["xy", "grid", "image"], category="grid")
class XYImageCollectInvocation(BaseInvocation):
    """XYImageCollect takes xItem, yItem and an Image and outputs it as an (x_item,y_item,image_name)array converted to json"""
    x_item: str = InputField(default='', description="The X item", input=Input.Connection)
    y_item: str = InputField(default='', description="The Y item", input=Input.Connection)
    image: ImageField = InputField(default=None, description="The image to turn into grids", input=Input.Connection)

    def invoke(self, context: InvocationContext) -> StringOutput:
        return StringOutput(value=json.dumps([self.y_item, self.x_item , self.image.image_name]))


@invocation("xyimage_grid", title="XYImage To Grid", tags=["xy", "grid", "image"], category="grid")
class XYImagesToGridInvocation(BaseInvocation):
    """Load a collection of xyimage types (json of (x_item,y_item,image_name)array) and create a gridimage of them"""
    xyimages: list[str] = InputField(default=[], description="The xyImage Collection", ui_type=UIType.Collection, input=Input.Connection)
    space: int = InputField(default=1, ge=0, description="The space to be added between images")
    scale_factor: Optional[float] = InputField(default=1.0, gt=0, description="The factor by which to scale the images")
    resample_mode:  PIL_RESAMPLING_MODES = InputField(default="bicubic", description="The resampling mode")
    background_color: ColorField = InputField(
        default=ColorField(r=0, g=0, b=0, a=255),
        description="The color to use as the background",
    )
    label_font_name: str = InputField(default="arial.ttf", description="Name of the font to use for labels")
    label_font_size: int = InputField(default=35, description="Size of the font to use for labels")
    top_label_height: int = InputField(default=50, description="Height of the top label area")
    left_label_width: int = InputField(default=100, description="Width of the left label area")
    label_font_color: ColorField = InputField(
        default=ColorField(r=255, g=255, b=255, a=255),
        description="The color to use for the label font",
    )
    
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

    def invoke(self, context: InvocationContext) -> ImageCollectionOutput:
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

        return ImageCollectionOutput(collection=grid_images)


@invocation("image_grid", title="Images To Grids", tags=["grid", "image"], category="grid")
class ImagesToGridsInvocation(BaseInvocation):
    """Load a collection of images and creat grid images from it and output a collection of genereated grid images"""
    images: list[ImageField] = InputField(default=[], description="The image collection to turn into grids", ui_type=UIType.ImageCollection, input=Input.Connection)
    columns: int = InputField(default=1, ge=1, description="The number of columns in each grid")
    rows: int = InputField(default=1, ge=1, description="The nuber of rows to have in each grid")
    space: int = InputField(default=1, ge=0, description="The space to be added between images")
    scale_factor: Optional[float] = InputField(default=1.0, gt=0, description="The factor by which to scale the images")
    resample_mode:  PIL_RESAMPLING_MODES = InputField(default="bicubic", description="The resampling mode")
    background_color: ColorField = InputField(
        default=ColorField(r=0, g=0, b=0, a=255),
        description="The color to use as the background",
    )

    def invoke(self, context: InvocationContext) -> ImageCollectionOutput:
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

        return ImageCollectionOutput(collection=grid_images)
