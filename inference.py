from fast_vertex_quality_inference.processing.data_manager import data_manager
from fast_vertex_quality_inference.processing.network_manager import network_manager


#### 
# INITIALISE NETWORK
###
rapidsim_PV_smearing_network = network_manager(
					network="/users/am13743/fast_vertexing_variables/inference/example/smearing_network.onnx", 
					config="/users/am13743/fast_vertexing_variables/inference/example/smearing_configs.pkl",
					transformers="/users/am13743/fast_vertexing_variables/inference/example/smearing_transformers.pkl",
					)

vertexing_network = network_manager(
                    network="/users/am13743/fast_vertexing_variables/inference/example/vertexting_network.onnx", 
                    config="/users/am13743/fast_vertexing_variables/inference/example/vertexing_configs.pkl",
                    transformers="/users/am13743/fast_vertexing_variables/inference/example/vertexing_transformers.pkl",
                    )



data = data_manager(
					tuple="/users/am13743/fast_vertexing_variables/inference/example/example.root", 
					particles_TRUEID=[321, 11, 11],
					mother_TRUEID=521,
					fully_reco=True,
					nPositive_missing_particles=0,
					nNegative_missing_particles=0,
					tree='DecayTree',
					particles=["K_plus", "e_plus", "e_minus"],
					mother = 'B_plus',
					intermediate = 'J_psi',
					)
data.process(
			output_tuple="/users/am13743/fast_vertexing_variables/inference/example/example_smeared.root", PV_smearing_network=rapidsim_PV_smearing_network,
			vertexing_network=vertexing_network,
			)




