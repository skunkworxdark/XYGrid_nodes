# 2023 skunkworxdark (https://github.com/skunkworxdark)

import json
import math
import re
import textwrap
from itertools import product
from pathlib import Path
from typing import Optional, Union

from PIL import Image, ImageDraw, ImageFont

import invokeai.assets.fonts as font_assets
from invokeai.app.invocations.baseinvocation import (
    BaseInvocation,
    BaseInvocationOutput,
    FieldDescriptions,
    Input,
    InputField,
    InvocationContext,
    OutputField,
    UIComponent,
    UIType,
    WithMetadata,
    WithWorkflow,
    invocation,
    invocation_output,
)
from invokeai.app.invocations.image import PIL_RESAMPLING_MAP, PIL_RESAMPLING_MODES
from invokeai.app.invocations.latent import SAMPLER_NAME_VALUES, SchedulerOutput
from invokeai.app.invocations.primitives import (
    BoardField,
    ColorField,
    FloatOutput,
    ImageCollectionOutput,
    ImageField,
    ImageOutput,
    IntegerOutput,
    StringCollectionOutput,
    StringOutput,
)
from invokeai.app.services.image_records.image_records_common import ImageCategory, ResourceOrigin


def is_all_numeric(array):
    pattern = r"^-?\d+(\.\d+)?$"
    return all(re.match(pattern, item) for item in array)


def sort_array(array):
    return sorted(array, key=float) if is_all_numeric(array) else sorted(array)


def is_all_numeric2(array, i):
    pattern = r"^-?\d+(\.\d+)?$"
    return all(re.match(pattern, item[i]) for item in array)


def sort_array2(array):
    isNum0 = is_all_numeric2(array, 0)
    isNum1 = is_all_numeric2(array, 1)

    return sorted(
        array,
        key=lambda x: (
            (float(x[1]) if isNum1 else x[1]),
            (float(x[0]) if isNum0 else x[0]),
        ),
    )


@invocation(
    "floats_to_strings",
    title="Floats To Strings",
    tags=["float", "string"],
    category="util",
    version="1.0.0",
)
class FloatsToStringsInvocation(BaseInvocation):
    """Converts a float or collections of floats to a collection of strings"""

    floats: Union[float, list[float]] = InputField(
        default_factory=list,
        description="float or collection of floats",
        input=Input.Connection,
        ui_type=UIType.FloatCollection,
    )

    def invoke(self, context: InvocationContext) -> StringCollectionOutput:
        if self.floats is None:
            raise Exception("No float or collection of floats provided")
        return StringCollectionOutput(
            collection=[str(x) for x in self.floats] if isinstance(self.floats, list) else [str(self.floats)]
        )


@invocation(
    "ints_to_strings",
    title="Ints To Strings",
    tags=["int", "string"],
    category="util",
    version="1.0.0",
)
class IntsToStringsInvocation(BaseInvocation):
    """Converts an integer or collection of integers to a collection of strings"""

    ints: Union[int, list[int]] = InputField(
        default_factory=list,
        description="int or collection of ints",
        input=Input.Connection,
        ui_type=UIType.IntegerCollection,
    )

    def invoke(self, context: InvocationContext) -> StringCollectionOutput:
        if self.ints is None:
            raise Exception("No int or collection of ints provided")
        return StringCollectionOutput(
            collection=[str(x) for x in self.ints] if isinstance(self.ints, list) else [str(self.ints)]
        )


@invocation(
    "csv_to_strings",
    title="CSV To Strings",
    tags=["xy", "grid", "csv"],
    category="util",
    version="1.0.0",
)
class CSVToStringsInvocation(BaseInvocation):
    """Converts a CSV string to a collection of strings"""

    csv: str = InputField(description="csv string")

    def invoke(self, context: InvocationContext) -> StringCollectionOutput:
        return StringCollectionOutput(collection=self.csv.split(",").rstrip())


@invocation(
    "string_to_float",
    title="String To Float",
    tags=["float", "string"],
    category="util",
    version="1.0.0",
)
class StringToFloatInvocation(BaseInvocation):
    """Converts a string to a float"""

    float_string: str = InputField(description="string containing a float to convert")

    def invoke(self, context: InvocationContext) -> FloatOutput:
        return FloatOutput(value=float(self.float_string))


@invocation(
    "string_to_int",
    title="String To Int",
    tags=["int"],
    category="util",
    version="1.0.0",
)
class StringToIntInvocation(BaseInvocation):
    """Converts a string to an integer"""

    int_string: str = InputField(description="string containing an integer to convert")

    def invoke(self, context: InvocationContext) -> IntegerOutput:
        return IntegerOutput(value=int(self.int_string))


@invocation(
    "string_to_scheduler",
    title="String To Scheduler",
    tags=["scheduler"],
    category="util",
    version="1.0.0",
)
class StringToSchedulerInvocation(BaseInvocation):
    """Converts a string to a scheduler"""

    # ddim,ddpm,deis,lms,lms_k,pndm,heun,heun_k,euler,euler_k,euler_a,kdpm_2,kdpm_2_a,dpmpp_2s,dpmpp_2s_k,dpmpp_2m,dpmpp_2m_k,dpmpp_2m_sde,dpmpp_2m_sde_k,dpmpp_sde,dpmpp_sde_k,unipc
    scheduler_string: str = InputField(description="string containing a scheduler to convert")

    def invoke(self, context: InvocationContext) -> SchedulerOutput:
        return SchedulerOutput(scheduler=self.scheduler_string.strip().lower())


@invocation_output("xy_collect_output")
class XYProductOutput(BaseInvocationOutput):
    """XYCProductOutput a collection that contains every combination of the input collections"""

    xy_item_collection: list[str] = OutputField(description="The XY Item collection")


@invocation(
    "xy_product",
    title="XY Product",
    tags=["xy", "grid", "collect"],
    category="grid",
    version="1.0.0",
)
class XYProductInvocation(BaseInvocation):
    """Takes X and Y string collections and outputs a XY Item collection with every combination of X and Y"""

    x_collection: list[str] = InputField(default_factory=list, description="The X collection")
    y_collection: list[str] = InputField(default_factory=list, description="The Y collection")

    def invoke(self, context: InvocationContext) -> XYProductOutput:
        combinations = list(product(self.x_collection, self.y_collection))
        json_combinations = [json.dumps(list(comb)) for comb in combinations]

        return XYProductOutput(xy_item_collection=json_combinations)


@invocation(
    "xy_product_csv",
    title="XY Product CSV",
    tags=["xy", "grid", "csv"],
    category="grid",
    version="1.0.0",
)
class XYProductCSVInvocation(BaseInvocation):
    """Converts X and Y CSV strings to an XY Item collection with every combination of X and Y"""

    x: str = InputField(description="x string", ui_component=UIComponent.Textarea)
    y: str = InputField(description="y string", ui_component=UIComponent.Textarea)

    def invoke(self, context: InvocationContext) -> XYProductOutput:
        x_list = self.x.split(",")
        y_list = self.y.split(",")
        combinations = list(product(x_list, y_list))
        json_combinations = [json.dumps(list(comb)) for comb in combinations]

        return XYProductOutput(xy_item_collection=json_combinations)


@invocation_output("xy_expand_output")
class XYExpandOutput(BaseInvocationOutput):
    """Two strings that are expanded from an XY Item"""

    x_item: str = OutputField(description="The X item")
    y_item: str = OutputField(description="The y item")


@invocation(
    "xy_expand",
    title="XY Expand",
    tags=["xy", "grid"],
    category="grid",
    version="1.0.0",
)
class XYExpandInvocation(BaseInvocation):
    """Takes an XY Item and outputs the X and Y as individual strings"""

    xy_item: str = InputField(description="The XY Item")

    def invoke(self, context: InvocationContext) -> XYExpandOutput:
        lst = json.loads(self.xy_item)
        x_item = str(lst[0]) if len(lst) > 0 else ""
        y_item = str(lst[1]) if len(lst) > 1 else ""
        return XYExpandOutput(x_item=x_item, y_item=y_item)


@invocation_output("xy_image_expand_output")
class XYImageExpandOutput(BaseInvocationOutput):
    """XY Image Expand Output"""

    x_item: str = OutputField(description="The X item")
    y_item: str = OutputField(description="The y item")
    image: ImageField = OutputField(description="The Image item")


@invocation(
    "xy_image_expand",
    title="XYImage Expand",
    tags=["xy", "grid"],
    category="grid",
    version="1.0.0",
)
class XYImageExpandInvocation(BaseInvocation):
    """Takes an XYImage item and outputs the X,Y and Image"""

    xyimage_item: str = InputField(description="The XYImage collection item")

    def invoke(self, context: InvocationContext) -> XYImageExpandOutput:
        lst = json.loads(self.xyimage_item)
        x_item = str(lst[0]) if len(lst) > 0 else ""
        y_item = str(lst[1]) if len(lst) > 1 else ""
        image = str(lst[2]) if len(lst) > 2 else ""

        return XYImageExpandOutput(x_item=x_item, y_item=y_item, image=ImageField(image_name=image))


@invocation(
    "xy_image_collect",
    title="XYImage Collect",
    tags=["xy", "grid", "image"],
    category="grid",
    version="1.0.0",
)
class XYImageCollectInvocation(BaseInvocation):
    """Takes xItem, yItem and an Image and outputs it as an XYImage Item (x_item,y_item,image_name)array converted to json"""

    x_item: str = InputField(description="The X item")
    y_item: str = InputField(description="The Y item")
    image: ImageField = InputField(description="The image to turn into grids")

    def invoke(self, context: InvocationContext) -> StringOutput:
        return StringOutput(value=json.dumps([self.x_item, self.y_item, self.image.image_name]))


@invocation(
    "xy_images_to_grid",
    title="XYImages To Grid",
    tags=["xy", "grid", "image"],
    category="grid",
    version="1.0.0",
)
class XYImagesToGridInvocation(BaseInvocation, WithWorkflow, WithMetadata):
    """Takes Collection of XYImages (json of (x_item,y_item,image_name)array), sorts the images into X,Y and creates a grid image with labels"""

    board: Optional[BoardField] = InputField(default=None, description=FieldDescriptions.board, input=Input.Direct)
    xyimages: list[str] = InputField(
        default_factory=list,
        description="The XYImage item Collection",
    )
    scale_factor: Optional[float] = InputField(
        default=1.0,
        gt=0,
        description="The factor by which to scale the images",
    )
    resample_mode: PIL_RESAMPLING_MODES = InputField(
        default="bicubic",
        description="The resampling mode",
    )
    left_label_width: int = InputField(
        default=100,
        description="Width of the left label area",
    )
    label_font_size: int = InputField(
        default=16,
        description="Size of the font to use for labels",
    )

    def invoke(self, context: InvocationContext) -> ImageOutput:
        left_label_width = self.left_label_width
        new_array = [json.loads(s) for s in self.xyimages]
        sorted_array = sort_array2(new_array)
        images = [context.services.images.get_pil_image(item[2]) for item in sorted_array]
        x_labels = sort_array(set([item[0] for item in sorted_array]))
        y_labels = sort_array(set([item[1] for item in sorted_array]))
        columns = len(x_labels)
        rows = len(y_labels)
        column_width = int(max([image.width for image in images]) * self.scale_factor)
        row_height = int(max([image.height for image in images]) * self.scale_factor)
        resample_mode = PIL_RESAMPLING_MAP[self.resample_mode]

        # Note - font may be found either in the repo if running an editable install, or in the venv if running a package install
        font_path = [x for x in [Path(y, "inter/Inter-Regular.ttf") for y in font_assets.__path__] if x.exists()]
        font = ImageFont.truetype(font_path[0].as_posix(), self.label_font_size)

        # Wrap labels
        x_labels_max_chars = int(column_width // (self.label_font_size * 0.6))
        y_labels_max_chars = int(left_label_width // (self.label_font_size * 0.6))
        x_labels_wrapped = [textwrap.wrap(x.rstrip(), x_labels_max_chars) for x in x_labels]
        y_labels_wrapped = [textwrap.wrap(y.rstrip(), y_labels_max_chars) for y in y_labels]

        # Calculate x_label_height based on the number of lines they are wrapped to
        top_label_heights = [
            len(label) * (font.getbbox("hg")[3] - font.getbbox("hg")[1] + 5) for label in x_labels_wrapped
        ]
        top_label_height = max(top_label_heights)

        # Calculate output image size
        output_width = column_width * columns + left_label_width
        output_height = row_height * rows + top_label_height

        # create output image and draw object
        output_image = Image.new("RGBA", (output_width, output_height), (255, 255, 255))
        draw = ImageDraw.Draw(output_image)

        # Draw images and labels into output_image
        y = top_label_height
        for iy in range(rows):
            iy_off = iy * columns
            x = left_label_width
            for ix in range(columns):
                image = images[iy_off + ix]
                if not self.scale_factor == 1.0:
                    image = image.resize(
                        (
                            int(column_width),
                            int(row_height),
                        ),
                        resample=resample_mode,
                    )
                output_image.paste(image, (x, y))

                # Add x label on the top row
                if iy == 0:
                    w, h = draw.multiline_textbbox((0, 0), "\n".join(x_labels_wrapped[ix]), font=font)[2:4]
                    draw.text(
                        (x + ((column_width - w) / 2), 0), "\n".join(x_labels_wrapped[ix]), fill="black", font=font
                    )

                # Add y label on the first column
                if ix == 0:
                    w, h = draw.multiline_textbbox((0, 0), "\n".join(y_labels_wrapped[iy]), font=font)[2:4]
                    draw.text((((left_label_width - w) / 2), y + ((row_height - h) / 2)), "\n".join(y_labels_wrapped[iy]), fill="black", font=font)

                x += column_width
            y += row_height

        image_dto = context.services.images.create(
            image=output_image,
            image_origin=ResourceOrigin.INTERNAL,
            image_category=ImageCategory.GENERAL,
            board_id=self.board.board_id if self.board else None,
            node_id=self.id,
            session_id=context.graph_execution_state_id,
            is_intermediate=self.is_intermediate,
            metadata=self.metadata,
            workflow=self.workflow,
        )

        return ImageOutput(
            image=ImageField(image_name=image_dto.image_name),
            width=image_dto.width,
            height=image_dto.height,
        )


@invocation(
    "images_to_grids",
    title="Images To Grids",
    tags=["grid", "image"],
    category="grid",
    version="1.0.0",
)
class ImagesToGridsInvocation(BaseInvocation, WithWorkflow, WithMetadata):
    """Takes a collection of images and outputs a collection of generated grid images"""

    board: Optional[BoardField] = InputField(default=None, description=FieldDescriptions.board, input=Input.Direct)
    images: list[ImageField] = InputField(
        default_factory=list,
        description="The image collection to turn into grids",
        ui_type=UIType.ImageCollection,
    )
    columns: int = InputField(
        default=1,
        ge=1,
        description="The number of columns in each grid",
    )
    rows: int = InputField(
        default=1,
        ge=1,
        description="The number of rows to have in each grid",
    )
    space: int = InputField(
        default=1,
        ge=0,
        description="The space to be added between images",
    )
    scale_factor: Optional[float] = InputField(
        default=1.0,
        gt=0,
        description="The factor by which to scale the images",
    )
    resample_mode: PIL_RESAMPLING_MODES = InputField(
        default="bicubic",
        description="The resampling mode",
    )
    background_color: ColorField = InputField(
        default=ColorField(r=0, g=0, b=0, a=255),
        description="The color to use as the background",
    )

    def invoke(self, context: InvocationContext) -> ImageCollectionOutput:
        """Convert an image list into a grids of images"""
        images = [context.services.images.get_pil_image(image.image_name) for image in self.images]
        column_width = int(max([image.width for image in images]) * self.scale_factor)
        row_height = int(max([image.height for image in images]) * self.scale_factor)
        output_width = column_width * self.columns + (self.space * (self.columns - 1))
        output_height = row_height * self.rows + (self.space * (self.rows - 1))
        resample_mode = PIL_RESAMPLING_MAP[self.resample_mode]

        column = 0
        row = 0
        x_offset = 0
        y_offset = 0
        output_image = Image.new("RGBA", (output_width, output_height), self.background_color.tuple())
        grid_images = []

        for image in images:
            if not self.scale_factor == 1.0:
                image = image.resize(
                    (
                        int(image.width * self.scale_factor),
                        int(image.width * self.scale_factor),
                    ),
                    resample=resample_mode,
                )

            output_image.paste(image, (x_offset, y_offset))

            column += 1
            x_offset += column_width + self.space
            if column >= self.columns:
                column = 0
                x_offset = 0
                y_offset += row_height + self.space
                row += 1

            if row >= self.rows:
                row = 0
                y_offset = 0
                image_dto = context.services.images.create(
                    image=output_image,
                    image_origin=ResourceOrigin.INTERNAL,
                    image_category=ImageCategory.GENERAL,
                    board_id=self.board.board_id if self.board else None,
                    node_id=self.id,
                    session_id=context.graph_execution_state_id,
                    is_intermediate=self.is_intermediate,
                    metadata=self.metadata,
                    workflow=self.workflow,
                )
                grid_images.append(ImageField(image_name=image_dto.image_name))
                output_image = Image.new(
                    "RGBA",
                    (output_width, output_height),
                    self.background_color.tuple(),
                )

        # if we are not on column and row 0 then we have a part done grid and need to save it
        if column > 0 or row > 0:
            image_dto = context.services.images.create(
                image=output_image,
                image_origin=ResourceOrigin.INTERNAL,
                image_category=ImageCategory.GENERAL,
                board_id=self.board.board_id if self.board else None,
                node_id=self.id,
                session_id=context.graph_execution_state_id,
                is_intermediate=self.is_intermediate,
                metadata=self.metadata,
                workflow=self.workflow,
            )
            grid_images.append(ImageField(image_name=image_dto.image_name))

        return ImageCollectionOutput(collection=grid_images)


@invocation(
    "image_to_xy_image_collection",
    title="Image To XYImage Collection",
    tags=["xy", "grid", "image"],
    category="grid",
    version="1.0.0",
)
class ImageToXYImageCollectionInvocation(BaseInvocation, WithWorkflow, WithMetadata):
    """Cuts an image up into columns and rows and outputs XYImage Collection"""

    # Inputs
    image: ImageField = InputField(description="The input image")
    columns: int = InputField(default=2, ge=2, le=256, description="The number of columns")
    rows: int = InputField(default=2, ge=2, le=256, description="The number of rows")

    def invoke(self, context: InvocationContext) -> StringCollectionOutput:
        img = context.services.images.get_pil_image(self.image.image_name)

        dy = img.height // self.rows
        dx = img.width // self.columns
        xyimages = []

        for iy in range(self.rows):
            for ix in range(self.columns):
                x = ix * dx
                y = iy * dy
                box = (x, y, x + dx, y + dy)
                img_crop = img.crop(box)
                image_dto = context.services.images.create(
                    image=img_crop,
                    image_origin=ResourceOrigin.INTERNAL,
                    image_category=ImageCategory.OTHER,
                    node_id=self.id,
                    session_id=context.graph_execution_state_id,
                    is_intermediate=self.is_intermediate,
                    metadata=self.metadata,
                    workflow=self.workflow,
                )
                xyimages.append(json.dumps([str(x), str(y), image_dto.image_name]))

        return StringCollectionOutput(collection=xyimages)


@invocation_output("image_to_xy_image_output")
class ImageToXYImageTilesOutput(BaseInvocationOutput):
    """Image To XYImage Tiles Output"""

    xyImages: list[str] = OutputField(description="The XyImage Collection")
    tile_x: int = OutputField(description="The tile x dimension")
    tile_y: int = OutputField(description="The tile y dimension")
    overlap: int = OutputField(description="The tile overlap size")
    scale: int = OutputField(description="The the scaling amount")


@invocation(
    "image_to_xy_image_tiles",
    title="Image To XYImage Tiles",
    tags=["xy", "tile", "image"],
    category="tile",
    version="1.0.0",
)
class ImageToXYImageTilesInvocation(BaseInvocation, WithWorkflow, WithMetadata):
    """Cuts an image up into overlapping tiles and outputs as an XYImage Collection (x,y is the final position of the tile)"""

    # Inputs
    image: ImageField = InputField(description="The input image")
    tile_x: int = InputField(default=512, ge=1, description="x resolution of generation tile")
    tile_y: int = InputField(default=512, ge=1, description="y resolution of generation tile")
    overlap: int = InputField(default=64, ge=0, description="tile overlap size")
    scale: int = InputField(default=2, ge=2, le=256, description="How much to scale to output")

    def invoke(self, context: InvocationContext) -> ImageToXYImageTilesOutput:
        img = context.services.images.get_pil_image(self.image.image_name)

        # scale down tile and overlap for the source image
        source_tile_x = self.tile_x // self.scale
        source_tile_y = self.tile_y // self.scale
        source_overlap = self.overlap // self.scale

        dx = source_tile_x - source_overlap
        dy = source_tile_y - source_overlap

        x_tiles = math.ceil(((img.width - source_overlap) / dx))
        y_tiles = math.ceil(((img.height - source_overlap) / dy))

        xyimages = []

        for iy in range(y_tiles):
            y1 = iy * dy
            y2 = y1 + source_tile_y
            if y1 > img.height:
                break  # avoid exceeding limits
            # if block exceed height then make it a full block starting at the bottom
            if y2 > img.height:
                y1 = img.height - source_tile_y
                y2 = img.height
            for ix in range(x_tiles):
                x1 = ix * dx
                x2 = x1 + source_tile_x
                if x1 > img.width:
                    break  # avoid exceeding limits
                # if block exceeds width then make it a full block starting at the right
                if x2 > img.width:
                    x1 = img.width - source_tile_x
                    x2 = img.width

                box = (x1, y1, x2, y2)
                img_crop = img.crop(box)
                image_dto = context.services.images.create(
                    image=img_crop,
                    image_origin=ResourceOrigin.INTERNAL,
                    image_category=ImageCategory.OTHER,
                    node_id=self.id,
                    session_id=context.graph_execution_state_id,
                    is_intermediate=self.is_intermediate,
                    metadata=self.metadata,
                    workflow=self.workflow,
                )
                xyimages.append(json.dumps([str(x1 * self.scale), str(y1 * self.scale), image_dto.image_name]))

        return ImageToXYImageTilesOutput(
            xyImages=xyimages,
            tile_x=self.tile_x,
            tile_y=self.tile_y,
            overlap=self.overlap,
            scale=self.scale,
        )


@invocation(
    "xy_image_tiles_to_image",
    title="XYImage Tiles To Image",
    tags=["xy", "tile", "image"],
    category="tile",
    version="1.0.0",
)
class XYImageTilesToImageInvocation(BaseInvocation, WithWorkflow, WithMetadata):
    """Takes a collection of XYImage Tiles (json of array(x_pos,y_pos,image_name)) and create an image from overlapping tiles"""

    board: Optional[BoardField] = InputField(default=None, description=FieldDescriptions.board, input=Input.Direct)
    xyimages: list[str] = InputField(
        default_factory=list,
        description="The xyImage Collection",
    )

    def invoke(self, context: InvocationContext) -> ImageOutput:
        new_array = [json.loads(s) for s in self.xyimages]
        sorted_array = sort_array2(new_array)
        images = [context.services.images.get_pil_image(item[2]) for item in sorted_array]
        x_names = sort_array(set([item[0] for item in sorted_array]))
        columns = len(x_names)
        y_names = sort_array(set([item[1] for item in sorted_array]))
        rows = len(y_names)
        tile_width = images[0].width
        tile_height = images[0].height

        max_x = int(max([float(x) for x in x_names]))
        max_y = int(max([float(y) for y in y_names]))

        output_width = max_x + tile_width
        output_height = max_y + tile_height

        output_image = Image.new("RGBA", (output_width, output_height))
        row_image = Image.new("RGBA", (output_width, tile_height))

        gy = Image.linear_gradient("L")
        gx = gy.rotate(90)

        # create the first row
        row_image.paste(images[0], (0, 0))
        next_x = tile_width
        for ix in range(1, columns):
            x = int(x_names[ix])
            row_image.paste(images[ix], (x, 0))
            overlap_x = next_x - x
            next_x += tile_width - overlap_x
            if overlap_x > 0:
                # blend X
                x_img1 = images[ix - 1].crop((tile_width - overlap_x, 0, tile_width, tile_height))
                x_img2 = images[ix].crop((0, 0, overlap_x, tile_height))
                x_img1.paste(x_img2, (0, 0), gx.resize((overlap_x, tile_height)))
                row_image.paste(x_img1, (x, 0))
        output_image.paste(row_image, (0, 0))

        # do the rest of the rows
        next_y = tile_height
        for iy in range(1, rows):
            # Add the first image for the row
            row_image_new = Image.new("RGBA", (output_width, tile_height))
            iy_off = iy * columns
            row_image_new.paste(images[iy_off], (0, 0))
            next_x = tile_width
            for ix in range(1, columns):
                x = int(x_names[ix])
                row_image_new.paste(images[iy_off + ix], (x, 0))
                overlap_x = next_x - x
                next_x += tile_width - overlap_x
                if overlap_x > 0:
                    # blend X overlap
                    x_img1 = images[(iy_off + ix) - 1].crop((tile_width - overlap_x, 0, tile_width, tile_height))
                    x_img2 = images[iy_off + ix].crop((0, 0, overlap_x, tile_height))
                    x_img1.paste(x_img2, (0, 0), gx.resize((overlap_x, tile_height)))
                    row_image_new.paste(x_img1, (x, 0))
            y = int(y_names[iy])
            output_image.paste(row_image_new, (0, y))
            overlap_y = next_y - y
            next_y += tile_width - overlap_y
            if overlap_y > 0:
                # blend y overlap
                y_img1 = row_image.crop((0, tile_height - overlap_y, output_width, tile_height))
                y_img2 = row_image_new.crop((0, 0, output_width, overlap_y))
                y_img1.paste(y_img2, (0, 0), gy.resize((output_width, overlap_y)))
                row_image_new.paste(y_img1, (0, 0))
                output_image.paste(row_image_new, (0, y))
            row_image = row_image_new

        # Save the image
        image_dto = context.services.images.create(
            image=output_image,
            image_origin=ResourceOrigin.INTERNAL,
            image_category=ImageCategory.GENERAL,
            board_id=self.board.board_id if self.board else None,
            node_id=self.id,
            session_id=context.graph_execution_state_id,
            is_intermediate=self.is_intermediate,
            metadata=self.metadata,
            workflow=self.workflow,
        )

        return ImageOutput(
            image=ImageField(image_name=image_dto.image_name),
            width=image_dto.width,
            height=image_dto.height,
        )
