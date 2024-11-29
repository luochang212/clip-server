import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'


import json
import triton_python_backend_utils as pb_utils

from transformers import CLIPProcessor


MODEL_NAME = 'openai/clip-vit-base-patch32'
MODEL_PATH = '/workspace'


class TritonPythonModel:
    def initialize(self, args):

        model_config = json.loads(args["model_config"])
        output0_config = pb_utils.get_output_config_by_name(model_config, "input_ids")
        output1_config = pb_utils.get_output_config_by_name(model_config, "attention_mask")
        self.output0_dtype = pb_utils.triton_string_to_numpy(output0_config["data_type"])
        self.output1_dtype = pb_utils.triton_string_to_numpy(output1_config["data_type"])

        self.processor = CLIPProcessor.from_pretrained(MODEL_NAME, cache_dir=MODEL_PATH)

    def execute(self, requests):
        output0_dtype = self.output0_dtype
        output1_dtype = self.output1_dtype

        responses = []
        for request in requests:
            in_0 = pb_utils.get_input_tensor_by_name(request, "text")
            texts = in_0.as_numpy().tolist()
            texts = [text.decode("utf-8") for text in texts]

            inputs = self.processor(text=texts,
                                    return_tensors="pt",
                                    padding=True,
                                    truncation=True)

            input_ids = inputs["input_ids"].numpy()
            attention_mask = inputs["attention_mask"].numpy()

            out_tensor_0 = pb_utils.Tensor("input_ids", input_ids.astype(output0_dtype))
            out_tensor_1 = pb_utils.Tensor("attention_mask", attention_mask.astype(output1_dtype))

            inference_response = pb_utils.InferenceResponse(output_tensors=[out_tensor_0, out_tensor_1])
            responses.append(inference_response)
        return responses

    def finalize(self):
        print("Cleaning up...")
