import os
import base64
import requests
import sklearn.metrics
import pandas as pd

from io import BytesIO
from PIL import Image


TRITON_DEFAULT_URL = 'http://localhost:8000'


def gen_abspath(
        directory: str,
        rel_path: str
) -> str:
    """
    Generate the absolute path by combining the given directory with a relative path.

    :param directory: The specified directory, which can be either an absolute or a relative path.
    :param rel_path: The relative path with respect to the 'dir'.
    :return: The resulting absolute path formed by concatenating the absolute directory
             and the relative path.
    """
    abs_dir = os.path.abspath(directory)
    return os.path.join(abs_dir, rel_path)


def read_csv(
    file_path: str,
    sep: str = ',',
    header: int = 0,
    on_bad_lines: str = 'warn',
    encoding: str = 'utf-8',
    dtype: dict = None,
    **kwargs
) -> pd.DataFrame:
    """
    Read a CSV file from the specified path.
    """
    return pd.read_csv(file_path,
                       header=header,
                       sep=sep,
                       on_bad_lines=on_bad_lines,
                       encoding=encoding,
                       dtype=dtype,
                       **kwargs)


def check_triton_health(triton_url=TRITON_DEFAULT_URL):
    """检查 triton 状态"""
    url = f'{triton_url}/v2/health/ready'
    response = requests.get(url)

    if response.status_code == 200:
        print("Triton server is ready.")
    else:
        print(f"Triton server is not ready. Status code: {response.status_code}")


def check_model_health(model_name,
                       triton_url=TRITON_DEFAULT_URL):
    """检查模型状态"""
    url = f'{triton_url}/v2/models/{model_name}/ready'
    response = requests.get(url)

    if response.status_code == 200:
        print(f"Model '{model_name}' is ready.")
    else:
        print(f"Model '{model_name}' is not ready. Status code: {response.status_code}")


def get_triton_meta_data(triton_url=TRITON_DEFAULT_URL):
    """检查 triton 元数据"""
    url = f'{triton_url}/v2'
    response = requests.get(url)
    
    if response.status_code == 200:
        metadata = response.json()
        return metadata
    else:
        return 'error'


def image_to_base64(image, format=None):
    if format is None:
        format = image.format

    buffered = BytesIO()
    image.save(buffered, format=format)  # format 图像格式，例如 "JPEG" 或 "PNG"
    img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_base64


def img_file_to_base64(image_path):
    try:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("ascii")
    except FileNotFoundError:
        return None


def base64_to_image(base64_string):
    return Image.open(BytesIO(base64.b64decode(base64_string)))


def simple_client(inputs,
                  model_name,
                  triton_url=TRITON_DEFAULT_URL,
                  model_version='1'):
    url = f"{triton_url}/v2/models/{model_name}/versions/{model_version}/infer"

    # 发送 POST 请求
    response = requests.post(url=url, json=inputs)
    if response.status_code != 200:
        raise Exception(f"Inference request failed with status code {response.status_code}: {response.text}")

    return response.json()


def eval_binary(
    y_true,
    y_label
):
    """
    Evaluate a binary classification task.
    """
    assert len(set(y_true)) <= 2
    assert len(set(y_label)) <= 2

    # Metrics that require the predicted labels (y_label)
    acc = sklearn.metrics.accuracy_score(y_true=y_true, y_pred=y_label)
    precision = sklearn.metrics.precision_score(y_true=y_true, y_pred=y_label)
    recall = sklearn.metrics.recall_score(y_true=y_true, y_pred=y_label)
    f1 = sklearn.metrics.f1_score(y_true=y_true, y_pred=y_label)
    cm = sklearn.metrics.confusion_matrix(y_true=y_true, y_pred=y_label)
    tn, fp, fn, tp = cm.ravel()

    print(f'accuracy: {acc:.5f}')
    print(f'precision: {precision:.5f}')
    print(f'recall: {recall:.5f}')
    print(f'f1_score: {f1:.5f}')
    print(f'True Positive (TP): {tp}')
    print(f'True Negative (TN): {tn}')
    print(f'False Positive (FP): {fp}')
    print(f'False Negative (FN): {fn}')
    print(f'confusion matrix:\n{cm}')


def eval_multiclass(
        y_true,
        y_label,
        average='weighted'
):
    """
    Evaluate a multiclass classification task.
    
    :param y_true: array-like of shape (n_samples,)
        True labels for the classification task.
    :param y_label: array-like of shape (n_samples,)
        Predicted labels for the classification task.
    :param average: str, default='weighted'
        Averaging method for calculating precision, recall, and F1 score.
        Options: ['micro', 'macro', 'weighted', None]

    :return: None
        Prints the following metrics to the console:
        - Accuracy
        - Precision (using the specified averaging method)
        - Recall (using the specified averaging method)
        - F1 Score (using the specified averaging method)
        - Confusion Matrix
        - Classification Report
    """
    # Ensure input is valid
    assert len(set(y_label)) > 2, "Use eval_binary for binary classification tasks."

    # Metrics that require the predicted labels (y_label)
    acc = sklearn.metrics.accuracy_score(y_true=y_true, y_pred=y_label)
    precision = sklearn.metrics.precision_score(y_true=y_true, y_pred=y_label, average=average)
    recall = sklearn.metrics.recall_score(y_true=y_true, y_pred=y_label, average=average)
    f1 = sklearn.metrics.f1_score(y_true=y_true, y_pred=y_label, average=average)
    cm = sklearn.metrics.confusion_matrix(y_true=y_true, y_pred=y_label)
    class_report = sklearn.metrics.classification_report(y_true=y_true, y_pred=y_label)
    
    print(f'accuracy: {acc:.5f}')
    print(f'precision ({average}): {precision:.5f}')
    print(f'recall ({average}): {recall:.5f}')
    print(f'f1_score ({average}): {f1:.5f}')
    print(f'confusion matrix:\n{cm}')
    print(f'classification report:\n{class_report}')


def eval_regression(
        y_true,
        y_pred
):
    """
    This function evaluates the performance of a regression model by calculating
    the Mean Absolute Error (MAE), Mean Squared Error (MSE), and R-squared (R2) score.

    :param y_true: An array of the true values.
    :param y_label: An array of the model's predictions.
    """
    mae = sklearn.metrics.mean_absolute_error(y_true, y_pred)
    mse = sklearn.metrics.mean_squared_error(y_true, y_pred)
    r2_score = sklearn.metrics.r2_score(y_true, y_pred)

    print(f'mae:      {mae:.5f}')
    print(f'mse:      {mse:.5f}')
    print(f'r2_score: {r2_score:.5f}')
