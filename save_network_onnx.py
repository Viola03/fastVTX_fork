from fast_vertex_quality.tools.config import read_definition, rd

from fast_vertex_quality.training_schemes.vertex_quality import vertex_quality_trainer
import fast_vertex_quality.tools.data_loader as data_loader

import pickle
import tensorflow as tf
import tf2onnx

rd.use_QuantileTransformer = False

use_intermediate = False
rd.include_dropout = True

# load_state = f"test_runs/20th_long_2000_lower_LR/networks/20th_long_2000_lower_LR"
load_state = f"test_runs/22nf_nomissmass_deeper/networks/22nf_nomissmass_deeper"

transformers = pickle.load(open(f"{load_state}_transfomers.pkl", "rb"))

print(f"Loading data...")
training_data_loader = data_loader.load_data(
    [],
    transformers=transformers,
    empty=True,
)

vertex_quality_trainer_obj = vertex_quality_trainer(
    training_data_loader,
    None,
    batch_size=64,
    load_config=load_state
)

vertex_quality_trainer_obj.load_state(tag=load_state)


# Extract the decoder (a Keras model)
decoder_model = vertex_quality_trainer_obj.decoder
print(decoder_model)

# Save the Keras model to ONNX format
onnx_model_path = "decoder_model.onnx"

# Convert Keras model to ONNX
# Since the model has multiple inputs, we need to provide the correct input specifications.
input_signature = [
    tf.TensorSpec(decoder_model.inputs[0].shape, tf.float32, name="input_latent"),
    tf.TensorSpec(decoder_model.inputs[1].shape, tf.float32, name="momentum_conditions"),
]
onnx_model, _ = tf2onnx.convert.from_keras(decoder_model, input_signature=input_signature, opset=13)

# Save the ONNX model to a file
with open(onnx_model_path, "wb") as f:
    f.write(onnx_model.SerializeToString())

print(f"Decoder model saved as {onnx_model_path}")
