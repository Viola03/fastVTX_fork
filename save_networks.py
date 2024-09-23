from fast_vertex_quality.tools.config import read_definition, rd

from fast_vertex_quality.training_schemes.vertex_quality import vertex_quality_trainer
from fast_vertex_quality.training_schemes.primary_vertex import primary_vertex_trainer

import fast_vertex_quality.tools.data_loader as data_loader

import pickle
import tensorflow as tf
import tf2onnx

def save_model(trainer_obj, onnx_model_path):

    decoder_model = trainer_obj.decoder

    input_signature = [
        tf.TensorSpec(decoder_model.inputs[0].shape, tf.float32, name="input_latent"),
        tf.TensorSpec(decoder_model.inputs[1].shape, tf.float32, name="momentum_conditions"),
    ]
    onnx_model, _ = tf2onnx.convert.from_keras(decoder_model, input_signature=input_signature, opset=13)

    with open(onnx_model_path, "wb") as f:
        f.write(onnx_model.SerializeToString())

    print(f"Decoder model saved as {onnx_model_path}")

def rapidsim_ify__branch_names(branches):

    rapidsim_ified__branches = []
    for branch in branches:
        
        original_branch = branch

        if "TRUEORIGINVERTEX" in branch:
            dim = branch[branch.index("TRUEORIGINVERTEX") + len("TRUEORIGINVERTEX_")]
            branch = branch.replace(f"TRUEORIGINVERTEX_{dim}",f"orig{dim}_TRUE")
        elif "ORIGINVERTEX" in branch:
            dim = branch[branch.index("ORIGINVERTEX") + len("ORIGINVERTEX_")]
            branch = branch.replace(f"ORIGINVERTEX_{dim}",f"orig{dim}")
        if "TRUEENDVERTEX" in branch:
            dim = branch[branch.index("TRUEENDVERTEX") + len("TRUEENDVERTEX_")]
            branch = branch.replace(f"TRUEENDVERTEX_{dim}",f"vtx{dim}_TRUE")
        elif "ENDVERTEX" in branch and "CHI2" not in branch:
            dim = branch[branch.index("ENDVERTEX") + len("ENDVERTEX_")]
            branch = branch.replace(f"ENDVERTEX_{dim}",f"vtx{dim}") 
        elif "_TRUEP" in branch:
            try:
                dim = branch[branch.index("_TRUEP") + len("_TRUEP_")]
            except:
                dim = ''
            branch = branch.replace(f"TRUEP_{dim}",f"P{dim}_TRUE") 

        branch = branch.replace('B_plus','MOTHER')
        branch = branch.replace('K_Kst','DAUGHTER1')
        branch = branch.replace('e_plus','DAUGHTER2')
        branch = branch.replace('e_minus','DAUGHTER3')
        branch = branch.replace('J_psi_1S','INTERMEDIATE')

        rapidsim_ified__branches.append(branch)     

    return rapidsim_ified__branches

def organise_and_save(save_location, load_state, trainer, tag):

    transformers = pickle.load(open(f"{load_state}_transfomers.pkl", "rb"))
    trainer_obj = trainer(load_config=load_state)
    trainer_obj.load_state(tag=load_state)
    save_model(trainer_obj, f"{tag}_model.onnx")

    conditions = trainer_obj.conditions
    targets = trainer_obj.targets
    latent_dim = trainer_obj.latent_dim

    transformers_branches = rapidsim_ify__branch_names(list(transformers.keys()))
    targets = rapidsim_ify__branch_names(targets)
    conditions = rapidsim_ify__branch_names(conditions)

    transformers = {transformers_branches[i]: value for i, (key, value) in enumerate(transformers.items())}
    with open(f"{save_location}/{tag}_transfomers.pkl", 'wb') as handle:
        pickle.dump(transformers, handle, protocol=pickle.HIGHEST_PROTOCOL)

    configs = {}
    configs["conditions"] = conditions
    configs["targets"] = targets
    configs["latent_dim"] = latent_dim
    with open(f"{save_location}/{tag}_configs.pkl", 'wb') as handle:
        pickle.dump(configs, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"\nRAPIDSIM conditions: {conditions}")
    print(f"\nRAPIDSIM targets: {targets}")
    print('\n\n')


save_location = "inference/example/models/"
rapidsim_smearing_state = f"networks/primary_vertex_job_new_processing2"
vertex_quality_state = f"test_runs/22nf_nomissmass_deeper/networks/22nf_nomissmass_deeper"

organise_and_save(save_location, rapidsim_smearing_state, primary_vertex_trainer, tag="smearing")
organise_and_save(save_location, vertex_quality_state, vertex_quality_trainer, tag="vertexing")
