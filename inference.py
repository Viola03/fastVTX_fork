from fast_vertex_quality_inference.processing.data_manager import tuple_manager
from fast_vertex_quality_inference.processing.network_manager import network_manager
import numpy as np

# config.verbose = True

#### 
# INITIALISE NETWORKS
###

rapidsim_PV_smearing_network = network_manager(
					network="inference/example/models/smearing_decoder_model.onnx", 
					config="inference/example/models/smearing_configs.pkl", 
					transformers="inference/example/models/smearing_transfomers.pkl", 
					)

vertexing_network = network_manager(
					network="inference/example/models/vertexing_decoder_model.onnx", 
					config="inference/example/models/vertexing_configs.pkl", 
					transformers="inference/example/models/vertexing_transfomers.pkl", 
                    )

vertexing_encoder = network_manager(
					network="inference/example/models/vertexing_encoder_model.onnx", 
					config="inference/example/models/vertexing_configs.pkl", 
					transformers="inference/example/models/vertexing_transfomers.pkl", 
                    )


#### 
# LOAD RAPIDSIM TUPLE
###


data_tuple = tuple_manager(
					tuple_location="inference/example/example.root",
					particles_TRUEID=[321, 11, 11],
					fully_reco=1,
					nPositive_missing_particles=0,
					nNegative_missing_particles=0,
					mother_particle_name="B_plus",
					intermediate_particle_name="J_psi",
					daughter_particle_names=["K_plus","e_plus","e_minus"],
					)


#### 
# SMEAR PV
###

smearing_conditions = data_tuple.get_branches(
					rapidsim_PV_smearing_network.conditions, 
					rapidsim_PV_smearing_network.Transformers, 
					numpy=True,
					)
smeared_PV_output = rapidsim_PV_smearing_network.query_network(
					['noise',smearing_conditions],
					)
data_tuple.smearPV(smeared_PV_output)



#### 
# COMPUTE CONDITIONS AND RUN VERTEXING NETWORK
###

data_tuple.append_conditional_information()
vertexing_conditions = data_tuple.get_branches(
					vertexing_network.conditions, 
					vertexing_network.Transformers, 
					numpy=True,
					)
vertexing_output = vertexing_network.query_network(
					['noise',vertexing_conditions],
					)
data_tuple.add_branches(
					vertexing_output
					)


#### 
# WRITE TUPLE
###

data_tuple.write(new_branches_to_keep=vertexing_network.targets)

