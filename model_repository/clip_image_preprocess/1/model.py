import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'


import json
import base64
import triton_python_backend_utils as pb_utils

from io import BytesIO
from PIL import Image
from transformers import CLIPProcessor


MODEL_NAME = 'openai/clip-vit-base-patch32'
MODEL_PATH = '/workspace'


class TritonPythonModel:
    def initialize(self, args):
        model_config = json.loads(args["model_config"])
        output0_config = pb_utils.get_output_config_by_name(model_config, "pixel_values")
        self.output0_dtype = pb_utils.triton_string_to_numpy(output0_config["data_type"])

        self.processor = CLIPProcessor.from_pretrained(MODEL_NAME, cache_dir=MODEL_PATH)

    def execute(self, requests):
        responses = []
        for request in requests:
            in_0 = pb_utils.get_input_tensor_by_name(request, "base64_images")
            base64_images = in_0.as_numpy().tolist()
            base64_images = [img.decode("utf-8") for img in base64_images]

            images = [self.base64_to_image(img) for img in base64_images]

            inputs = self.processor(images=images, return_tensors="pt").to("cpu")
            pixel_values = inputs["pixel_values"].numpy()

            out_tensor_0 = pb_utils.Tensor("pixel_values", pixel_values.astype(self.output0_dtype))

            inference_response = pb_utils.InferenceResponse(output_tensors=[out_tensor_0])
            responses.append(inference_response)
        return responses

    def finalize(self):
        print("Cleaning up...")

    @staticmethod
    def base64_to_image(base64_string):
        return Image.open(BytesIO(base64.b64decode(base64_string)))
