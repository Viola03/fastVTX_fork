from fast_vertex_quality.tools.config import read_definition, rd

from fast_vertex_quality.training_schemes.vertex_quality import vertex_quality_trainer
from fast_vertex_quality.training_schemes.primary_vertex import primary_vertex_trainer

import fast_vertex_quality.tools.data_loader as data_loader

import pickle
import tensorflow as tf
import tf2onnx


def save_model(trainer_obj, onnx_model_path):

    # Extract the decoder (a Keras model)
    decoder_model = trainer_obj.decoder

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



#######
rapidsim_smearing_state = f"networks/primary_vertex_job_new_processing2"

load_state = rapidsim_smearing_state
vertex_quality_transformers = pickle.load(open(f"{load_state}_transfomers.pkl", "rb"))
primary_vertex_trainer_obj = primary_vertex_trainer(load_config=load_state)
primary_vertex_trainer_obj.load_state(tag=load_state)
save_model(primary_vertex_trainer_obj, "smearing_decoder_model.onnx")


#######
vertex_quality_state = f"test_runs/22nf_nomissmass_deeper/networks/22nf_nomissmass_deeper"

load_state = vertex_quality_state
vertex_quality_transformers = pickle.load(open(f"{load_state}_transfomers.pkl", "rb"))
vertex_quality_trainer_obj = vertex_quality_trainer(load_config=load_state)
vertex_quality_trainer_obj.load_state(tag=load_state)
save_model(vertex_quality_trainer_obj, "vertexing_decoder_model.onnx")

