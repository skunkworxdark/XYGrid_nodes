# 2024 skunkworxdark (https://github.com/skunkworxdark)

import csv
import io
import json
import math
import os
import re
import textwrap
from itertools import product
from typing import Any, Literal, Optional, Union

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps
from PIL.Image import Image as PILImageType

from invokeai.app.invocations.constants import LATENT_SCALE_FACTOR
from invokeai.app.invocations.flux_model_loader import FluxModelLoaderInvocation, FluxModelLoaderOutput
from invokeai.app.invocations.image import PIL_RESAMPLING_MAP, PIL_RESAMPLING_MODES
from invokeai.app.invocations.model import MainModelLoaderInvocation
from invokeai.app.invocations.sd3_model_loader import Sd3ModelLoaderInvocation, Sd3ModelLoaderOutput
from invokeai.app.invocations.sdxl import SDXLModelLoaderInvocation, SDXLModelLoaderOutput
from invokeai.invocation_api import (
    SCHEDULER_NAME_VALUES,
    BaseInvocation,
    BaseInvocationOutput,
    ColorField,
    FieldDescriptions,
    FloatOutput,
    ImageCollectionOutput,
    ImageField,
    ImageOutput,
    Input,
    InputField,
    IntegerOutput,
    InvocationContext,
    LatentsField,
    LatentsOutput,
    ModelIdentifierField,
    ModelLoaderOutput,
    OutputField,
    SchedulerOutput,
    StringCollectionOutput,
    StringOutput,
    UIComponent,
    UIType,
    WithBoard,
    WithMetadata,
    invocation,
    invocation_output,
)

_downsampling_factor = LATENT_SCALE_FACTOR

# numeric pattern
# - ^s* - start of line then any whitespace
# - [-+]? - optional -+ at a start
# - \s* - any whitespace before number
# - (\d+(\.\d*)?|\.\d+) - number.(number)optional|.number
# - \s* - any whitespace after the number
# - %?\s*$ - optional % and any whitespace after
num_pattern = re.compile(r"^\s*[-+]?\s*(\d+(\.\d*)?|\.\d+)\s*%?\s*$")


def is_numeric(s: str) -> bool:
    """checks if a string is numeric ignoring leading + trailing % and any whitespace."""

    return bool(num_pattern.match(s))


def prep_num(s: str) -> str:
    """removes all + or % or whitespace"""

    return "".join(s.split()).replace("%", "").replace("+", "")


def is_all_numeric(array: list[str]) -> bool:
    """returns if all elements in an array are numeric or not"""

    return all(is_numeric(item) for item in array)


def sort_array(array: list[str]) -> list[str]:
    """sort array of str but if they are all numeric then it will sort the as numbers"""

    return sorted(array, key=lambda x: float(prep_num(x))) if is_all_numeric(array) else sorted(array)


def is_all_numeric2(array: list[tuple[str, str, str]], i: int) -> bool:
    """returns if all elements in a 2D array index i are numeric or not"""

    return all(is_numeric(item[i]) for item in array)


def sort_array2(array: list[tuple[str, str, str]]) -> list[tuple[str, str, str]]:
    """sort 2D array of str but if they are all numeric then it will sort them as numeric
    specifically it will sort them index 1,0 because it is expecting x,y.
    This is to ensure it is in the right order for XY grid processing"""

    isNum0 = is_all_numeric2(array, 0)
    isNum1 = is_all_numeric2(array, 1)

    return sorted(
        array,
        key=lambda x: (
            (float(prep_num(x[1])) if isNum1 else x[1]),
            (float(prep_num(x[0])) if isNum0 else x[0]),
        ),
    )


def shift(arr: np.ndarray[Any, Any], num: int, fill_value: float = 255.0) -> np.ndarray[Any, Any]:
    result = np.full_like(arr, fill_value)
    if num > 0:
        result[num:] = arr[:-num]
    elif num < 0:
        result[:num] = arr[-num:]
    else:
        result[:] = arr
    return result


BLEND_MODES = Literal[
    "Linear",
    "seam-grad",
    "seam-sobel1",
    "seam-sobel3",
    "seam-sobel5",
    "seam-sobel7",
    "seam-scharr",
]


def get_seam_line(
    i1: PILImageType,
    i2: PILImageType,
    rotate: bool,
    gutter: int,
    search_size: int = 1,
    blend_mode: BLEND_MODES = "seam-grad",
) -> PILImageType:
    ia1 = np.array(i1.convert("RGB")) / 255.0
    # BT.601 luminance conversion
    lc = np.array([0.2989, 0.5870, 0.1140])
    if i1.mode != "L":
        ia1 = np.tensordot(ia1, lc, axes=1)

    ia2 = np.array(i2.convert("RGB")) / 255.0
    if i2.mode != "L":
        ia2 = np.tensordot(ia2, lc, axes=1)

    # calc difference between images
    ia = ia2 - ia1

    if rotate:
        ia = np.rot90(ia, 1)

    # array is y by x
    max_y, max_x = ia.shape
    max_x -= gutter
    min_x = gutter

    if blend_mode == "seam-sobel1":
        # Use Sobel operator for energy calculation
        # gx = cv2.Sobel(ia, cv2.CV_64F, 1, 0, ksize=1)
        gx = cv2.Sobel(ia, -1, 1, 0, ksize=1)
        gy = cv2.Sobel(ia, -1, 0, 1, ksize=1)
        energy = np.hypot(gx, gy)
    elif blend_mode == "seam-sobel3":
        # Use Sobel operator for energy calculation
        gx = cv2.Sobel(ia, -1, 1, 0, ksize=3)
        gy = cv2.Sobel(ia, -1, 0, 1, ksize=3)
        energy = np.hypot(gx, gy)
    elif blend_mode == "seam-sobel5":
        # Use Sobel operator for energy calculation
        gx = cv2.Sobel(ia, -1, 1, 0, ksize=5)
        gy = cv2.Sobel(ia, -1, 0, 1, ksize=5)
        energy = np.hypot(gx, gy)
    elif blend_mode == "seam-sobel7":
        # Use Sobel operator for energy calculation
        gx = cv2.Sobel(ia, -1, 1, 0, ksize=7)
        gy = cv2.Sobel(ia, -1, 0, 1, ksize=7)
        energy = np.hypot(gx, gy)
    elif blend_mode == "seam-scharr":
        # Use Sobel operator for energy calculation
        gx = cv2.Scharr(ia, -1, 1, 0)
        gy = cv2.Scharr(ia, -1, 0, 1)
        energy = np.hypot(gx, gy)
    elif blend_mode == "seam-grad":
        # Calc the energy in the difference
        energy = np.abs(np.gradient(ia, axis=0)) + np.abs(np.gradient(ia, axis=1))
    else:
        raise ValueError(f"Unsupported blend mode: '{blend_mode}'.")

    ie = Image.fromarray((energy * 255.0).astype("uint8"))
    print(f"energy{ie.size}")

    res = np.copy(energy)

    for y in range(1, max_y):
        row = res[y, :]
        rowl = shift(row, -1)
        rowr = shift(row, 1)
        res[y, :] = res[y - 1, :] + np.min([row, rowl, rowr], axis=0)

    # create an array max_y long
    lowest_energy_line = np.empty([max_y], dtype="uint16")
    lowest_energy_line[max_y - 1] = np.argmin(res[max_y - 1, min_x : max_x - 1])

    for ypos in range(max_y - 2, -1, -1):
        lowest_pos = lowest_energy_line[ypos + 1]
        lpos = lowest_pos - search_size
        rpos = lowest_pos + search_size
        lpos = np.clip(lpos, min_x, max_x - 1)
        rpos = np.clip(rpos, min_x, max_x - 1)
        lowest_energy_line[ypos] = np.argmin(energy[ypos, lpos : rpos + 1]) + lpos

    mask = np.zeros_like(ia)

    for ypos in range(0, max_y):
        to_fill = lowest_energy_line[ypos]
        mask[ypos, 0:to_fill] = 1

    if rotate:
        mask = np.rot90(mask, 3)

    image = Image.fromarray((mask * 255.0).astype("uint8"))

    return image


def seam_mask(
    i1: PILImageType,
    i2: PILImageType,
    rotate: bool,
    blur_size: int,
    search_size: int = 1,
    blend_mode: BLEND_MODES = "seam-grad",
) -> PILImageType:
    seam = get_seam_line(i1, i2, rotate, blur_size + 1, search_size=search_size, blend_mode=blend_mode)
    #    blur = ImageFilter.GaussianBlur(float(blur_size))
    blur = ImageFilter.BoxBlur(float(blur_size))
    mask = seam.filter(blur)
    mask = ImageOps.invert(mask)
    return mask


def csv_line_to_list(csv_string: str) -> list[str]:
    """Converts the first line of a CSV into a list of strings"""
    with io.StringIO(csv_string) as input:
        reader = csv.reader(input)
        return next(reader)


@invocation(
    "main_model_loader_input",
    title="Main Model Input",
    tags=["model"],
    category="model",
    version="1.1.0",
)
class MainModelLoaderInputInvocation(MainModelLoaderInvocation):
    """Loads a main model from an input, outputting its submodels."""

    model: ModelIdentifierField = InputField(description=FieldDescriptions.main_model, ui_type=UIType.MainModel)

    def invoke(self, context: InvocationContext) -> ModelLoaderOutput:
        return super().invoke(context)


@invocation(
    "sdxl_model_loader_input",
    title="SDXL Main Model Input",
    tags=["model", "sdxl"],
    category="model",
    version="1.1.0",
)
class SDXLModelLoaderInputInvocation(SDXLModelLoaderInvocation):
    """Loads a sdxl model from an input, outputting its submodels."""

    model: ModelIdentifierField = InputField(description=FieldDescriptions.main_model, ui_type=UIType.SDXLMainModel)

    def invoke(self, context: InvocationContext) -> SDXLModelLoaderOutput:
        return super().invoke(context)


@invocation(
    "flux_model_loader_input",
    title="FLUX Main Model Input",
    tags=["model", "flux"],
    category="model",
    version="1.0.0",
)
class FluxModelLoaderInputInvocation(FluxModelLoaderInvocation):
    """Loads a flux model from an input, outputting its submodels."""

    model: ModelIdentifierField = InputField(description=FieldDescriptions.flux_model, ui_type=UIType.FluxMainModel)

    t5_encoder_model: ModelIdentifierField = InputField(
        description=FieldDescriptions.t5_encoder, ui_type=UIType.T5EncoderModel, title="T5 Encoder"
    )

    clip_embed_model: ModelIdentifierField = InputField(
        description=FieldDescriptions.clip_embed_model,
        ui_type=UIType.CLIPEmbedModel,
        title="CLIP Embed",
    )

    vae_model: ModelIdentifierField = InputField(
        description=FieldDescriptions.vae_model, ui_type=UIType.FluxVAEModel, title="VAE"
    )

    def invoke(self, context: InvocationContext) -> FluxModelLoaderOutput:
        return super().invoke(context)


@invocation(
    "sd3_model_loader_input",
    title="SD3 Main Model Input",
    tags=["model", "sd3"],
    category="model",
    version="1.0.0",
)
class SD3ModelLoaderInputInvocation(Sd3ModelLoaderInvocation):
    """Loads a sd3 model from an input, outputting its submodels."""

    model: ModelIdentifierField = InputField(description=FieldDescriptions.sd3_model, ui_type=UIType.SD3MainModel)

    t5_encoder_model: Optional[ModelIdentifierField] = InputField(
        description=FieldDescriptions.t5_encoder,
        ui_type=UIType.T5EncoderModel,
        title="T5 Encoder",
        default=None,
    )

    clip_l_model: Optional[ModelIdentifierField] = InputField(
        description=FieldDescriptions.clip_embed_model,
        ui_type=UIType.CLIPLEmbedModel,
        title="CLIP L Encoder",
        default=None,
    )

    clip_g_model: Optional[ModelIdentifierField] = InputField(
        description=FieldDescriptions.clip_g_model,
        ui_type=UIType.CLIPGEmbedModel,
        title="CLIP G Encoder",
        default=None,
    )

    vae_model: Optional[ModelIdentifierField] = InputField(
        description=FieldDescriptions.vae_model, ui_type=UIType.VAEModel, title="VAE", default=None
    )

    def invoke(self, context: InvocationContext) -> Sd3ModelLoaderOutput:
        return super().invoke(context)


@invocation_output("string_to_main_model_output")
class StringToMainModelOutput(BaseInvocationOutput):
    """String to main model output"""

    model: ModelIdentifierField = OutputField(description=FieldDescriptions.main_model, title="Model")
    name: str = OutputField(description="Model Name", title="Name")


@invocation(
    "string_to_main_model",
    title="String To Main Model",
    tags=["model"],
    category="model",
    version="1.1.0",
)
class StringToMainModelInvocation(BaseInvocation):
    """Loads a main model from a json string, outputting its submodels."""

    model_string: str = InputField(description="string containing a Model to convert")

    def invoke(self, context: InvocationContext) -> StringToMainModelOutput:
        model = ModelIdentifierField.model_validate_json(self.model_string)
        return StringToMainModelOutput(model=model, name=f"{model.base}: {model.name}")


@invocation_output("string_to_sdxl_model_output")
class StringToSDXLModelOutput(BaseInvocationOutput):
    """String to SDXL main model output"""

    model: ModelIdentifierField = OutputField(
        description=FieldDescriptions.main_model, title="Model", ui_type=UIType.SDXLMainModel
    )
    name: str = OutputField(description="Model Name", title="Name")


@invocation(
    "string_to_sdxl_model",
    title="String To SDXL Main Model",
    tags=["model", "sdxl"],
    category="model",
    version="1.1.0",
)
class StringToSDXLModelInvocation(BaseInvocation):
    """Loads a SDXL model from a json string, outputting its submodels."""

    model_string: str = InputField(description="string containing a Model to convert")

    def invoke(self, context: InvocationContext) -> StringToSDXLModelOutput:
        model = ModelIdentifierField.model_validate_json(self.model_string)
        return StringToSDXLModelOutput(model=model, name=f"{model.base}: {model.name}")


@invocation_output("string_to_lora_output")
class StringToLoraOutput(BaseInvocationOutput):
    """String to Lora model output"""

    model: ModelIdentifierField = OutputField(
        description=FieldDescriptions.lora_model, title="Model", ui_type=UIType.LoRAModel
    )
    name: str = OutputField(description="Model Name", title="Name")


@invocation(
    "string_to_lora",
    title="String To LoRA",
    tags=["model", "lora"],
    category="model",
    version="1.0.0",
)
class StringToLoraInvocation(BaseInvocation):
    """Loads a lora from a json string, outputting its submodels."""

    model_string: str = InputField(description="string containing a Model to convert")

    def invoke(self, context: InvocationContext) -> StringToLoraOutput:
        model = ModelIdentifierField.model_validate_json(self.model_string)
        return StringToLoraOutput(model=model, name=f"{model.base}: {model.name}")


@invocation_output("string_to_model_output")
class StringToModelOutput(BaseInvocationOutput):
    """String to model output"""

    model: ModelIdentifierField = OutputField(description="Model identifier", title="Model")
    name: str = OutputField(description="Model Name", title="Name")


@invocation(
    "string_to_model",
    title="String To Model",
    tags=["model"],
    category="model",
    version="1.0.0",
)
class StringToModelInvocation(BaseInvocation):
    """Loads a model from a json string, outputting its submodels."""

    model_string: str = InputField(description="string containing a Model to convert")

    def invoke(self, context: InvocationContext) -> StringToModelOutput:
        model = ModelIdentifierField.model_validate_json(self.model_string)
        return StringToModelOutput(model=model, name=f"{model.base}: {model.name}")


@invocation(
    "main_model_to_string",
    title="Main Model To String",
    tags=["model", "picker"],
    category="model",
    version="1.1.0",
)
class MainModelToStringInvocation(BaseInvocation):
    """Converts a Main Model to a JSONString"""

    model: ModelIdentifierField = InputField(description=FieldDescriptions.main_model, ui_type=UIType.MainModel)

    def invoke(self, context: InvocationContext) -> StringOutput:
        return StringOutput(value=self.model.model_dump_json())


@invocation(
    "sdxl_model_to_string",
    title="SDXL Model To String",
    tags=["model", "sdxl"],
    category="model",
    version="1.1.0",
)
class SDXLModelToStringInvocation(BaseInvocation):
    """Converts an SDXL Model to a JSONString"""

    model: ModelIdentifierField = InputField(
        description=FieldDescriptions.sdxl_main_model, ui_type=UIType.SDXLMainModel
    )

    def invoke(self, context: InvocationContext) -> StringOutput:
        return StringOutput(value=self.model.model_dump_json())


@invocation(
    "flux_model_to_string",
    title="Flux Model To String",
    tags=["model", "flux"],
    category="model",
    version="1.0.0",
)
class FluxModelToStringInvocation(BaseInvocation):
    """Converts an Flux Model to a JSONString"""

    model: ModelIdentifierField = InputField(description=FieldDescriptions.flux_model, ui_type=UIType.FluxMainModel)

    def invoke(self, context: InvocationContext) -> StringOutput:
        return StringOutput(value=self.model.model_dump_json())


@invocation(
    "sd3_model_to_string",
    title="SD3 Model To String",
    tags=["model", "sd3"],
    category="model",
    version="1.0.0",
)
class Sd3ModelToStringInvocation(BaseInvocation):
    """Converts an SD3 Model to a JSONString"""

    model: ModelIdentifierField = InputField(description=FieldDescriptions.sd3_model, ui_type=UIType.SD3MainModel)

    def invoke(self, context: InvocationContext) -> StringOutput:
        return StringOutput(value=self.model.model_dump_json())


@invocation(
    "lora_to_string",
    title="LoRA To String",
    tags=["model", "lora"],
    category="model",
    version="1.0.0",
)
class LoraToStringInvocation(BaseInvocation):
    """Converts an lora Model to a JSONString"""

    model: ModelIdentifierField = InputField(description=FieldDescriptions.lora_model, ui_type=UIType.LoRAModel)

    def invoke(self, context: InvocationContext) -> StringOutput:
        return StringOutput(value=self.model.model_dump_json())


@invocation(
    "model_to_string",
    title="Model To String",
    tags=["model", "lora"],
    category="model",
    version="1.0.0",
)
class ModelToStringInvocation(BaseInvocation):
    """Converts an Model to a JSONString"""

    model: ModelIdentifierField = InputField(description="The model to select", title="Model")

    def invoke(self, context: InvocationContext) -> StringOutput:
        return StringOutput(value=self.model.model_dump_json())


@invocation(
    "scheduler_to_string",
    title="Scheduler To String",
    tags=["scheduler"],
    category="model",
    version="1.0.1",
)
class SchedulerToStringInvocation(BaseInvocation):
    """Converts a Scheduler to a string"""

    scheduler: SCHEDULER_NAME_VALUES = InputField(
        default="euler",
        description=FieldDescriptions.scheduler,
        ui_type=UIType.Scheduler,
    )

    def invoke(self, context: InvocationContext) -> StringOutput:
        return StringOutput(value=self.scheduler)


@invocation(
    "floats_to_strings",
    title="Floats To Strings",
    tags=["float", "string"],
    category="util",
    version="1.0.1",
)
class FloatsToStringsInvocation(BaseInvocation):
    """Converts a float or collections of floats to a collection of strings"""

    floats: Union[float, list[float]] = InputField(
        default=[],
        description="float or collection of floats",
        input=Input.Connection,
    )

    def invoke(self, context: InvocationContext) -> StringCollectionOutput:
        return StringCollectionOutput(
            collection=[str(x) for x in self.floats] if isinstance(self.floats, list) else [str(self.floats)]
        )


@invocation(
    "ints_to_strings",
    title="Ints To Strings",
    tags=["int", "string"],
    category="util",
    version="1.1.1",
)
class IntsToStringsInvocation(BaseInvocation):
    """Converts an integer or collection of integers to a collection of strings"""

    ints: Union[int, list[int]] = InputField(
        default=[],
        description="int or collection of ints",
        input=Input.Connection,
    )

    def invoke(self, context: InvocationContext) -> StringCollectionOutput:
        return StringCollectionOutput(
            collection=[str(x) for x in self.ints] if isinstance(self.ints, list) else [str(self.ints)]
        )


@invocation(
    "csv_to_strings",
    title="CSV To Strings",
    tags=["xy", "grid", "csv"],
    category="util",
    version="1.1.0",
)
class CSVToStringsInvocation(BaseInvocation):
    """Converts a CSV string to a collection of strings"""

    csv_string: str = InputField(description="csv string")

    def invoke(self, context: InvocationContext) -> StringCollectionOutput:
        return StringCollectionOutput(collection=csv_line_to_list(self.csv_string))


@invocation(
    "string_to_float",
    title="String To Float",
    tags=["float", "string"],
    category="util",
    version="1.0.1",
)
class StringToFloatInvocation(BaseInvocation):
    """Converts a string to a float"""

    float_string: str = InputField(description="string containing a float to convert")

    def invoke(self, context: InvocationContext) -> FloatOutput:
        return FloatOutput(value=float(prep_num(self.float_string)))


@invocation(
    "percent_to_float",
    title="Percent To Float",
    tags=["float", "percentage"],
    category="string",
    version="1.0.1",
)
class PercentToFloatInvocation(BaseInvocation):
    """Converts a string to a float and divides it by 100."""

    text: str = InputField(
        title="Text",
        description="Input text",
    )

    def invoke(self, context: InvocationContext) -> FloatOutput:
        output = float(prep_num(self.text)) / 100
        return FloatOutput(value=output)


@invocation(
    "string_to_int",
    title="String To Int",
    tags=["int"],
    category="util",
    version="1.0.1",
)
class StringToIntInvocation(BaseInvocation):
    """Converts a string to an integer"""

    int_string: str = InputField(description="string containing an integer to convert")

    def invoke(self, context: InvocationContext) -> IntegerOutput:
        return IntegerOutput(value=int(prep_num(self.int_string)))


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
    version="1.1.0",
)
class XYProductInvocation(BaseInvocation):
    """Takes X and Y string collections and outputs a XY Item collection with every combination of X and Y"""

    x_collection: list[str] = InputField(default=[], description="The X collection")
    y_collection: list[str] = InputField(default=[], description="The Y collection")

    def invoke(self, context: InvocationContext) -> XYProductOutput:
        combinations = list(product(self.x_collection, self.y_collection))
        json_combinations = [json.dumps(list(comb)) for comb in combinations]

        return XYProductOutput(xy_item_collection=json_combinations)


@invocation(
    "xy_product_csv",
    title="XY Product CSV",
    tags=["xy", "grid", "csv"],
    category="grid",
    version="1.0.1",
)
class XYProductCSVInvocation(BaseInvocation):
    """Converts X and Y CSV strings to an XY Item collection with every combination of X and Y"""

    x: str = InputField(description="x string", ui_component=UIComponent.Textarea)
    y: str = InputField(description="y string", ui_component=UIComponent.Textarea)

    def invoke(self, context: InvocationContext) -> XYProductOutput:
        x_list = csv_line_to_list(self.x)
        y_list = csv_line_to_list(self.y)
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
    width: int = OutputField(description="The width of the image in pixels")
    height: int = OutputField(description="The height of the image in pixels")


@invocation(
    "xy_image_expand",
    title="XYImage Expand",
    tags=["xy", "grid"],
    category="grid",
    version="1.1.1",
)
class XYImageExpandInvocation(BaseInvocation):
    """Takes an XYImage item and outputs the XItem,YItem, Image, width & height"""

    xyimage_item: str = InputField(description="The XYImage collection item")

    def invoke(self, context: InvocationContext) -> XYImageExpandOutput:
        lst = json.loads(self.xyimage_item)
        x_item = str(lst[0]) if len(lst) > 0 else ""
        y_item = str(lst[1]) if len(lst) > 1 else ""
        image_name = str(lst[2]) if len(lst) > 2 else ""
        image = context.images.get_pil(image_name)

        return XYImageExpandOutput(
            x_item=x_item,
            y_item=y_item,
            image=ImageField(image_name=image_name),
            width=image.width,
            height=image.height,
        )


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
    version="1.3.1",
)
class XYImagesToGridInvocation(BaseInvocation, WithMetadata, WithBoard):
    """Takes Collection of XYImages (json of (x_item,y_item,image_name)array), sorts the images into X,Y and creates a grid image with labels"""

    #    board: Optional[BoardField] = InputField(default=None, description=FieldDescriptions.board, input=Input.Direct)
    xyimages: list[str] = InputField(
        default=[],
        description="The XYImage item Collection",
    )
    scale_factor: float = InputField(
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
        images = [context.images.get_pil(item[2]) for item in sorted_array]
        x_labels = sort_array(list({item[0] for item in sorted_array}))
        y_labels = sort_array(list({item[1] for item in sorted_array}))
        columns = len(x_labels)
        rows = len(y_labels)
        column_width = int(max([image.width for image in images]) * self.scale_factor)
        row_height = int(max([image.height for image in images]) * self.scale_factor)
        resample_mode = PIL_RESAMPLING_MAP[self.resample_mode]

        import invokeai.assets.fonts.inter as fp

        font_path = os.path.join(fp.__path__[0], "Inter-Regular.ttf")
        assert os.path.exists(font_path), f"Font file not found: {font_path}"
        font = ImageFont.truetype(font_path, self.label_font_size)

        # Wrap labels
        x_labels_max_chars = int(column_width // (self.label_font_size * 0.6))
        y_labels_max_chars = int(left_label_width // (self.label_font_size * 0.6))
        x_labels_wrapped = [textwrap.wrap(x.rstrip(), x_labels_max_chars) for x in x_labels]
        y_labels_wrapped = [textwrap.wrap(y.rstrip(), y_labels_max_chars) for y in y_labels]

        # Calculate x_label_height based on the number of lines they are wrapped to
        font_height = sum(font.getmetrics())
        max_lines = max(len(label) for label in x_labels_wrapped)
        top_label_height = (max_lines * font_height) + 5

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
                            int(image.width * self.scale_factor),
                            int(image.height * self.scale_factor),
                        ),
                        resample=resample_mode,
                    )
                output_image.paste(image, (x, y))

                # Add x label on the top row
                if iy == 0:
                    w, h = draw.multiline_textbbox((0, 0), "\n".join(x_labels_wrapped[ix]), font=font)[2:4]
                    draw.text(
                        (x + ((column_width - w) / 2), 0),
                        "\n".join(x_labels_wrapped[ix]),
                        fill="black",
                        font=font,
                    )

                # Add y label on the first column
                if ix == 0:
                    w, h = draw.multiline_textbbox((0, 0), "\n".join(y_labels_wrapped[iy]), font=font)[2:4]
                    draw.text(
                        (((left_label_width - w) / 2), y + ((row_height - h) / 2)),
                        "\n".join(y_labels_wrapped[iy]),
                        fill="black",
                        font=font,
                    )

                x += column_width
            y += row_height

        image_dto = context.images.save(output_image)

        return ImageOutput.build(image_dto)


@invocation(
    "images_to_grids",
    title="Images To Grids",
    tags=["grid", "image"],
    category="grid",
    version="1.3.0",
)
class ImagesToGridsInvocation(BaseInvocation, WithMetadata, WithBoard):
    """Takes a collection of images and outputs a collection of generated grid images"""

    #    board: Optional[BoardField] = InputField(default=None, description=FieldDescriptions.board, input=Input.Direct)
    images: list[ImageField] = InputField(
        default=[],
        description="The image collection to turn into grids",
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
    scale_factor: float = InputField(
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
        images = [context.images.get_pil(image.image_name) for image in self.images]
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
        grid_images: list[ImageField] = []

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
                image_dto = context.images.save(output_image)
                grid_images.append(ImageField(image_name=image_dto.image_name))
                output_image = Image.new(
                    "RGBA",
                    (output_width, output_height),
                    self.background_color.tuple(),
                )

        # if we are not on column and row 0 then we have a part done grid and need to save it
        if column > 0 or row > 0:
            image_dto = context.images.save(output_image)
            grid_images.append(ImageField(image_name=image_dto.image_name))

        return ImageCollectionOutput(collection=grid_images)


@invocation(
    "image_to_xy_image_collection",
    title="Image To XYImage Collection",
    tags=["xy", "grid", "image"],
    category="grid",
    version="1.1.1",
)
class ImageToXYImageCollectionInvocation(BaseInvocation, WithMetadata):
    """Cuts an image up into columns and rows and outputs XYImage Collection"""

    # Inputs
    image: ImageField = InputField(description="The input image")
    columns: int = InputField(default=2, ge=2, le=256, description="The number of columns")
    rows: int = InputField(default=2, ge=2, le=256, description="The number of rows")

    def invoke(self, context: InvocationContext) -> StringCollectionOutput:
        img = context.images.get_pil(self.image.image_name)

        dy = img.height // self.rows
        dx = img.width // self.columns
        xyimages: list[str] = []

        for iy in range(self.rows):
            for ix in range(self.columns):
                x = ix * dx
                y = iy * dy
                box = (x, y, x + dx, y + dy)
                img_crop = img.crop(box)
                image_dto = context.images.save(img_crop)
                xyimages.append(json.dumps([str(x), str(y), image_dto.image_name]))

        return StringCollectionOutput(collection=xyimages)


@invocation_output("tiles_output")
class TilesOutput(BaseInvocationOutput):
    """Tiles Output"""

    tiles: list[str] = OutputField(description="Tiles Collection")


@invocation(
    "default_xy_tile_generator",
    title="Default XYImage Tile Generator",
    tags=["xy", "tile"],
    category="tile",
    version="1.1.1",
)
class DefaultXYTileGenerator(BaseInvocation):
    """Cuts up an image into overlapping tiles and outputs a string representation of the tiles to use"""

    # Inputs
    image: ImageField = InputField(description="The input image")
    tile_width: int = InputField(
        default=576,
        ge=1,
        multiple_of=_downsampling_factor,
        description="x resolution of generation tile (must be a multiple of 8)",
    )
    tile_height: int = InputField(
        default=576,
        ge=1,
        multiple_of=_downsampling_factor,
        description="y resolution of generation tile (must be a multiple of 8)",
    )
    overlap: int = InputField(
        default=128,
        ge=0,
        multiple_of=_downsampling_factor,
        description="tile overlap size (must be a multiple of 8)",
    )
    adjust_tile_size: bool = InputField(
        default=True,
        description="adjust tile size to account for overlap",
    )

    def invoke(self, context: InvocationContext) -> TilesOutput:
        img = context.images.get_pil(self.image.image_name)

        if self.adjust_tile_size:
            tiles_x = img.width // self.tile_width
            tiles_y = img.height // self.tile_height
            self.tile_width = (img.width + self.overlap * (tiles_x - 1)) // tiles_x
            self.tile_height = (img.height + self.overlap * (tiles_y - 1)) // tiles_y

        if img.width < self.tile_width:
            self.tile_width = img.width

        if img.height < self.tile_height:
            self.tile_height = img.height

        dx = self.tile_width - self.overlap
        dy = self.tile_height - self.overlap

        x_tiles = math.ceil(((img.width - self.overlap) / dx))
        y_tiles = math.ceil(((img.height - self.overlap) / dy))

        xytiles: list[str] = []
        xytiles.append(json.dumps(str(self.image.image_name)))

        for iy in range(y_tiles):
            y1 = iy * dy
            y2 = y1 + self.tile_height
            if y1 > img.height:
                break  # avoid exceeding limits
            # if block exceed height then make it a full block starting at the bottom
            if y2 > img.height:
                y1 = img.height - self.tile_height
                y2 = img.height
            for ix in range(x_tiles):
                x1 = ix * dx
                x2 = x1 + self.tile_width
                if x1 > img.width:
                    break  # avoid exceeding limits
                # if block exceeds width then make it a full block starting at the right
                if x2 > img.width:
                    x1 = img.width - self.tile_width
                    x2 = img.width

                xytiles.append(json.dumps([str(x1), str(y1), str(x2), str(y2)]))

        return TilesOutput(tiles=xytiles)


@invocation(
    "minimum_overlap_xy_tile_generator",
    title="Minimum Overlap XYImage Tile Generator",
    tags=["xy", "tile"],
    category="tile",
    version="1.1.1",
)
class MinimumOverlapXYTileGenerator(BaseInvocation):
    """Cuts up an image into overlapping tiles and outputs a string representation of the tiles to use, taking the
    input overlap as a minimum"""

    # Inputs
    image: ImageField = InputField(description="The input image")
    tile_width: int = InputField(
        default=576,
        ge=1,
        multiple_of=_downsampling_factor,
        description="x resolution of generation tile (must be a multiple of 8)",
    )
    tile_height: int = InputField(
        default=576,
        ge=1,
        multiple_of=_downsampling_factor,
        description="y resolution of generation tile (must be a multiple of 8)",
    )
    min_overlap: int = InputField(
        default=128,
        ge=0,
        multiple_of=_downsampling_factor,
        description="minimum tile overlap size (must be a multiple of 8)",
    )
    round_to_8: bool = InputField(
        default=False,
        description="Round outputs down to the nearest 8 (for pulling from a large noise field)",
    )

    def invoke(self, context: InvocationContext) -> TilesOutput:
        img = context.images.get_pil(self.image.image_name)

        if img.width < self.tile_width:
            self.tile_width = img.width

        if img.height < self.tile_height:
            self.tile_height = img.height

        num_tiles_w = (
            math.ceil((img.width - self.min_overlap) / (self.tile_width - self.min_overlap))
            if self.tile_width < img.width
            else 1
        )
        num_tiles_h = (
            math.ceil((img.height - self.min_overlap) / (self.tile_height - self.min_overlap))
            if self.tile_height < img.height
            else 1
        )

        xytiles: list[str] = []
        xytiles.append(json.dumps(str(self.image.image_name)))

        for yiter in range(num_tiles_h):
            y1 = (yiter * (img.height - self.tile_height)) // (num_tiles_h - 1) if num_tiles_h > 1 else 0
            if self.round_to_8:
                y1 = 8 * (y1 // 8)
            y2 = y1 + self.tile_height
            for xiter in range(num_tiles_w):
                x1 = (xiter * (img.width - self.tile_width)) // (num_tiles_w - 1) if num_tiles_w > 1 else 0
                if self.round_to_8:
                    x1 = 8 * (x1 // 8)
                x2 = x1 + self.tile_width

                xytiles.append(json.dumps([str(x1), str(y1), str(x2), str(y2)]))

        return TilesOutput(tiles=xytiles)


@invocation(
    "even_split_xy_tile_generator",
    title="Even Split XYImage Tile Generator",
    tags=["xy", "tile"],
    category="tile",
    version="1.1.1",
)
class EvenSplitXYTileGenerator(BaseInvocation):
    """Cuts up an image into a number of even sized tiles with the overlap been a percentage of the tile size and outputs a string representation of the tiles to use"""

    # Inputs
    image: ImageField = InputField(description="The input image")
    num_x_tiles: int = InputField(
        default=2,
        ge=1,
        description="Number of tiles to divide image into on the x axis",
    )
    num_y_tiles: int = InputField(
        default=2,
        ge=1,
        description="Number of tiles to divide image into on the y axis",
    )
    overlap: float = InputField(
        default=0.25,
        ge=0,
        lt=1,
        description="Overlap amount of tile size (0-1)",
    )

    def invoke(self, context: InvocationContext) -> TilesOutput:
        img = context.images.get_pil(self.image.image_name)

        # Ensure tile size is divisible by 8
        if img.width % 8 != 0 or img.height % 8 != 0:
            raise ValueError(f"image size (({img.width}, {img.height})) must be divisible by 8")

        # Calculate the overlap size based on the percentage
        overlap_x = int((img.width / self.num_x_tiles) * self.overlap)
        overlap_y = int((img.height / self.num_y_tiles) * self.overlap)

        # Adjust overlap to be divisible by 8
        if overlap_x % 8 != 0:
            overlap_x = 8 * ((overlap_x // 8) + 1)
        if overlap_y % 8 != 0:
            overlap_y = 8 * ((overlap_y // 8) + 1)

        # Calculate the tile size based on the number of tiles and overlap
        tile_size_x = (img.width + overlap_x * (self.num_x_tiles - 1)) // self.num_x_tiles
        tile_size_y = (img.height + overlap_y * (self.num_y_tiles - 1)) // self.num_y_tiles

        # Ensure tile size is divisible by 8
        if tile_size_x % 8 != 0:
            tile_size_x = 8 * ((tile_size_x) // 8)
        if tile_size_y % 8 != 0:
            tile_size_y = 8 * ((tile_size_y) // 8)

        xytiles: list[str] = []
        xytiles.append(json.dumps(str(self.image.image_name)))

        for yi in range(self.num_y_tiles):
            for xi in range(self.num_x_tiles):
                # Calculate the top left coordinate of each tile
                top_left_x = xi * (tile_size_x - overlap_x)
                top_left_y = yi * (tile_size_y - overlap_y)

                # Calculate the bottom right coordinate of each tile
                bottom_right_x = min(top_left_x + tile_size_x, img.width)
                bottom_right_y = min(top_left_y + tile_size_y, img.height)

                # Adjust the last tiles in each row and column to fit exactly within the width and height of the image
                if xi == self.num_x_tiles - 1:
                    bottom_right_x = img.width
                if yi == self.num_y_tiles - 1:
                    bottom_right_y = img.height

                # Append the coordinates to the list
                xytiles.append(json.dumps([str(top_left_x), str(top_left_y), str(bottom_right_x), str(bottom_right_y)]))

        return TilesOutput(tiles=xytiles)


@invocation_output("image_to_xy_image_output")
class ImageToXYImageTilesOutput(BaseInvocationOutput):
    """Image To XYImage Tiles Output"""

    xyImages: list[str] = OutputField(description="The XYImage Collection")


@invocation(
    "image_to_xy_image_tiles",
    title="Image To XYImage Tiles",
    tags=["xy", "tile", "image"],
    category="tile",
    version="1.3.1",
)
class ImageToXYImageTilesInvocation(BaseInvocation):
    """Cuts an image up into overlapping tiles and outputs as an XYImage Collection (x,y is the final position of the tile)"""

    # Inputs
    tiles: list[str] = InputField(default=[], description="The list of tiles")

    def invoke(self, context: InvocationContext) -> ImageToXYImageTilesOutput:
        tiles = self.tiles.copy()

        image_name = json.loads(tiles.pop(0))
        img = context.images.get_pil(image_name)

        xyimages: list[str] = []

        for item in tiles:
            x1, y1, x2, y2 = [int(i) for i in json.loads(item)]

            box = (x1, y1, x2, y2)
            img_crop = img.crop(box)
            image_dto = context.images.save(img_crop)
            xyimages.append(json.dumps([str(x1), str(y1), image_dto.image_name]))

        return ImageToXYImageTilesOutput(xyImages=xyimages)


@invocation(
    "xy_image_tiles_to_image",
    title="XYImage Tiles To Image",
    tags=["xy", "tile", "image"],
    category="tile",
    version="1.3.0",
)
class XYImageTilesToImageInvocation(BaseInvocation, WithMetadata, WithBoard):
    """Takes a collection of XYImage Tiles (json of array(x_pos,y_pos,image_name)) and create an image from overlapping tiles"""

    #    board: Optional[BoardField] = InputField(default=None, description=FieldDescriptions.board, input=Input.Direct)
    xyimages: list[str] = InputField(
        default=[],
        description="The xyImage Collection",
    )
    blend_mode: BLEND_MODES = InputField(
        default="seam-grad",
        description="Seam blending type Linear or Smart",
        input=Input.Direct,
    )
    blur_size: int = InputField(
        default=16,
        ge=0,
        description="Size of the blur & Gutter to use with Smart Seam",
    )
    search_size: int = InputField(
        default=1,
        ge=1,
        description="Seam search size in pixels 1-4 are sensible sizes",
        input=Input.Direct,
    )

    def invoke(self, context: InvocationContext) -> ImageOutput:
        new_array = [json.loads(s) for s in self.xyimages]
        sorted_array = sort_array2(new_array)
        images = [context.images.get_pil(item[2]) for item in sorted_array]
        x_coords = sort_array(list({item[0] for item in sorted_array}))
        columns = len(x_coords)
        y_coords = sort_array(list({item[1] for item in sorted_array}))
        rows = len(y_coords)

        # use the last tile position and the tiles image size to calculate the output size
        output_width = images[-1].width + int(x_coords[-1])
        output_height = images[-1].height + int(y_coords[-1])

        output_image = Image.new("RGBA", (output_width, output_height))
        row_image = Image.new("RGBA", (output_width, images[0].height))

        # create linear gradient masks
        gy = Image.linear_gradient("L")
        gx = gy.rotate(90)

        # create the first row
        row_image.paste(images[0], (0, 0))
        next_x = images[0].width
        for ix in range(1, columns):
            x = int(x_coords[ix])
            row_image.paste(images[ix], (x, 0))
            overlap_x = next_x - x
            next_x += images[ix].width - overlap_x
            if overlap_x > 0:
                # blend X
                x_img1 = images[ix - 1].crop(
                    (images[ix - 1].width - overlap_x, 0, images[ix - 1].width, images[ix - 1].height)
                )
                x_img2 = images[ix].crop((0, 0, overlap_x, images[ix].height))
                if self.blend_mode == "Linear":
                    x_img1.paste(x_img2, (0, 0), gx.resize((overlap_x, images[ix].height)))
                else:
                    mask = seam_mask(
                        x_img1, x_img2, False, self.blur_size, search_size=self.search_size, blend_mode=self.blend_mode
                    )
                    x_img1.paste(x_img2, (0, 0), mask)
                row_image.paste(x_img1, (x, 0))
        output_image.paste(row_image, (0, 0))

        # do the rest of the rows
        next_y = images[0].height
        for iy in range(1, rows):
            # Add the first image for the row
            iy_off = iy * columns
            row_image_new = Image.new("RGBA", (output_width, images[iy_off].height))
            row_image_new.paste(images[iy_off], (0, 0))
            next_x = images[iy_off].width
            for ix in range(1, columns):
                x = int(x_coords[ix])
                row_image_new.paste(images[iy_off + ix], (x, 0))
                overlap_x = next_x - x
                next_x += images[iy_off + ix].width - overlap_x
                if overlap_x > 0:
                    # blend X overlap
                    x_img1 = images[(iy_off + ix) - 1].crop(
                        (
                            images[(iy_off + ix) - 1].width - overlap_x,
                            0,
                            images[(iy_off + ix) - 1].width,
                            images[(iy_off + ix) - 1].height,
                        )
                    )
                    x_img2 = images[iy_off + ix].crop((0, 0, overlap_x, images[iy_off + ix].height))
                    if self.blend_mode == "Linear":
                        x_img1.paste(x_img2, (0, 0), gx.resize((overlap_x, images[iy_off + ix].height)))
                    else:
                        mask = seam_mask(
                            x_img1,
                            x_img2,
                            False,
                            self.blur_size,
                            search_size=self.search_size,
                            blend_mode=self.blend_mode,
                        )
                        x_img1.paste(x_img2, (0, 0), mask)
                    row_image_new.paste(x_img1, (x, 0))
            y = int(y_coords[iy])
            output_image.paste(row_image_new, (0, y))
            overlap_y = next_y - y
            next_y += images[iy_off].height - overlap_y
            if overlap_y > 0:
                # blend y overlap
                y_img1 = row_image.crop((0, row_image.height - overlap_y, output_width, row_image.height))
                y_img2 = row_image_new.crop((0, 0, output_width, overlap_y))
                if self.blend_mode == "Linear":
                    y_img1.paste(y_img2, (0, 0), gy.resize((output_width, overlap_y)))
                else:
                    mask = seam_mask(
                        y_img1, y_img2, True, self.blur_size, search_size=self.search_size, blend_mode=self.blend_mode
                    )
                    y_img1.paste(y_img2, (0, 0), mask)
                row_image_new.paste(y_img1, (0, 0))
                output_image.paste(row_image_new, (0, y))
            row_image = row_image_new

        # Save the image
        image_dto = context.images.save(output_image)

        return ImageOutput.build(image_dto)


@invocation(
    "lcrop",
    title="Crop Latents",
    tags=["latents", "crop"],
    category="latents",
    version="1.0.1",
)
class CropLatentsInvocation(BaseInvocation):
    """Crops latents"""

    latents: LatentsField = InputField(
        description=FieldDescriptions.latents,
        input=Input.Connection,
    )
    width: int = InputField(
        ge=64,
        multiple_of=_downsampling_factor,
        description=FieldDescriptions.width,
    )
    height: int = InputField(
        ge=64,
        multiple_of=_downsampling_factor,
        description=FieldDescriptions.width,
    )
    x_offset: int = InputField(
        ge=0,
        multiple_of=_downsampling_factor,
        description="x-coordinate",
    )
    y_offset: int = InputField(
        ge=0,
        multiple_of=_downsampling_factor,
        description="y-coordinate",
    )

    def invoke(self, context: InvocationContext) -> LatentsOutput:
        latents = context.tensors.load(self.latents.latents_name)

        x1 = self.x_offset // _downsampling_factor
        y1 = self.y_offset // _downsampling_factor
        x2 = x1 + (self.width // _downsampling_factor)
        y2 = y1 + (self.height // _downsampling_factor)

        cropped_latents = latents[:, :, y1:y2, x1:x2]

        # resized_latents = resized_latents.to("cpu")

        name = context.tensors.save(tensor=cropped_latents)

        return LatentsOutput.build(latents_name=name, latents=cropped_latents)
