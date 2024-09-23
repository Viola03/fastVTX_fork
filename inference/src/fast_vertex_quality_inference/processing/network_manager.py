import numpy as np
import uproot
import pickle
import onnxruntime as ort
from fast_vertex_quality_inference.processing.transformers import Transformer as Transformer

class network_manager:

    def __init__(self, 
                network,
                config,
                transformers,
                ):
        
        config = pickle.load(open(config, "rb"))

        self.conditions = config[5]
        self.targets = config[6]

        self.Transformers = pickle.load(open(transformers, "rb"))

        self.branches = self.conditions + self.targets

        print(f"\n########\nStarting up ONNX InferenceSession for:\n{network}\n")
        self.session = ort.InferenceSession(network)

        # Check model inputs
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        print(f"\tModel Input Names: {self.input_names}")
        input_shapes = [inp.shape[1] for inp in self.session.get_inputs()]
        print(f"\tModel Input Dimensions: {input_shapes}")
        for idx, name in enumerate(self.input_names):
            if 'latent' in name:
                self.latent_dim = input_shapes[idx]
            
        print('\n')
        # Check model outputs
        self.output_names = [out.name for out in self.session.get_outputs()]
        print(f"\tModel Output Names: {self.output_names}")
        output_shapes = [out.shape[1] for out in self.session.get_outputs()]
        print(f"\tModel Output Dimensions: {output_shapes}")
        print('\n')

        print(f"\tCondition branches: {self.conditions}")
        print(f"\tTarget branches: {self.targets}")
        print('\n########\n')


        