
import json
import pickle
import numpy as np
import pandas as pd
import os
import joblib
from azureml.core.model import Model

from inference_schema.schema_decorators import input_schema, output_schema
from inference_schema.parameter_types.numpy_parameter_type import NumpyParameterType
from inference_schema.parameter_types.pandas_parameter_type import PandasParameterType


def init():
    global model
    # Replace filename if needed.
    path = os.getenv('AZUREML_MODEL_DIR') 
    model_path = os.path.join(path, 'smepml_multinomialmb_model.pkl')
    # Deserialize the model file back into a sklearn model.
    model = joblib.load(model_path)

input_sample = np.array(["Aturdimiento Deposiciones oscuras o con sangre Perdida del conocimiento en casos graves Vomitos de sangre"])
# This is an integer type sample. Use the data type that reflects the expected result.
output_sample = np.array([0])

@input_schema('data', NumpyParameterType(input_sample))
@output_schema(NumpyParameterType(output_sample))
def run(data):
    try:
        df = pd.DataFrame(model.predict_proba(data), columns=model.classes_)

        d = df.apply(lambda c: str(c[0]).strip('[]')).to_dict()

        ds = {k: v for k, v in sorted(d.items(), key=lambda item: item[1], reverse=True)} 

        dt = {k: ds[k] for k in list(ds)[:5]}

        result = str(dt)
    # You can return any data type, as long as it can be serialized by JSON.
        return result
    except Exception as e:
        error = str(e)
        return error
