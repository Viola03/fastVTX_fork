from fast_vertex_quality.tools.config import read_definition, rd

# from fast_vertex_quality.training_schemes.vertex_quality import vertex_quality_trainer
from fast_vertex_quality.training_schemes.primary_vertex import primary_vertex_trainer
import fast_vertex_quality.tools.data_loader as data_loader

import pickle
import tensorflow as tf
import tf2onnx
import numpy as np

rd.use_QuantileTransformer = False

use_intermediate = False
rd.include_dropout = True

load_state = f"networks/primary_vertex_job_new_processing2"

transformers = pickle.load(open(f"{load_state}_transfomers.pkl", "rb"))

print(f"Loading data...")
training_data_loader = data_loader.load_data(
    [],
    transformers=transformers,
    empty=True,
)

primary_vertex_trainer_obj = primary_vertex_trainer(
    training_data_loader,
    None,
    batch_size=64,
    load_config=load_state
)

# mother = 'MOTHER'
# conditions = [
#     f"{mother}_TRUEP",
#     f"{mother}_TRUEP_T",
#     f"{mother}_TRUEP_X",
#     f"{mother}_TRUEP_Y",
#     f"{mother}_TRUEP_Z",
# ]

# targets = [
#     f"{mother}_TRUE_FD",
#     f"{mother}_TRUEORIGINVERTEX_X",
#     f"{mother}_TRUEORIGINVERTEX_Y",
#     f"{mother}_TRUEORIGINVERTEX_Z",
# ]
# rd.latent = 2 

# primary_vertex_trainer_obj = primary_vertex_trainer(
# 		None,
# 		conditions=conditions,
# 		targets=targets,
# 		beta=float(rd.beta),
# 		latent_dim=rd.latent,
# 		batch_size=256,
# 		D_architecture=[1000,2000,1000],
# 		G_architecture=[1000,2000,1000],
# 		network_option='VAE',
# 	)

primary_vertex_trainer_obj.load_state(tag=load_state)


# [[ 0.22490545  0.85108083  0.12945804  0.01033347]
#  [ 0.2452658   0.86108065  0.1304835  -0.63626313]
#  [ 0.22377439  0.8536781   0.11894263  0.03956172]
#  ...
#  [-0.23329645  0.8496253   0.13111873  0.06913963]
#  [-0.19868723  0.8487214   0.13110112  0.0413713 ]
#  [-0.16821958  0.8487333   0.1299447  -0.17910454]]

# events_gen = np.load('conditions_smearing.npy')
# gen_noise = np.load('noise_smearing.npy')

# images = np.squeeze(primary_vertex_trainer_obj.decoder.predict([gen_noise, events_gen]))
# print(images)


# Extract the decoder (a Keras model)
decoder_model = primary_vertex_trainer_obj.decoder
print(decoder_model)

# Save the Keras model to ONNX format
onnx_model_path = "smearing_decoder_model.onnx"

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




# from fast_vertex_quality_inference.processing.network_manager import network_manager
# rapidsim_PV_smearing_network = network_manager(
#             network=onnx_model_path, 
#             config="/users/am13743/fast_vertexing_variables/inference/example/smearing_configs.pkl",
#             transformers="/users/am13743/fast_vertexing_variables/inference/example/smearing_transformers.pkl",
#             )
# input_data = {
#     rapidsim_PV_smearing_network.input_names[0]: gen_noise.astype(np.float32),
#     rapidsim_PV_smearing_network.input_names[1]: events_gen.astype(np.float32)
# }
# images2 = np.squeeze(rapidsim_PV_smearing_network.session.run(rapidsim_PV_smearing_network.output_names, input_data))

# print(images2)
