import importlib

__attributes = {
    "BasicTrainer": "basic",
    "SparseStructureVaeTrainer": "vae.sparse_structure_vae",
    "VcwSLatVaeGaussianTrainer": "vae.structured_latent_vae_gaussian_vcw",
    "SLatVaeGaussianTrainer": "vae.structured_latent_vae_gaussian",
    "SLatVaeRadianceFieldDecoderTrainer": "vae.structured_latent_vae_rf_dec",
    "SLatVaeMeshDecoderTrainer": "vae.structured_latent_vae_mesh_dec",
    "FlowMatchingTrainer": "flow_matching.flow_matching",
    "FlowMatchingCFGTrainer": "flow_matching.flow_matching",
    "TextConditionedFlowMatchingCFGTrainer": "flow_matching.flow_matching",
    "ImageConditionedFlowMatchingCFGTrainer": "flow_matching.flow_matching",
    "SparseFlowMatchingTrainer": "flow_matching.sparse_flow_matching",
    "SparseFlowMatchingCFGTrainer": "flow_matching.sparse_flow_matching",
    "TextConditionedSparseFlowMatchingCFGTrainer": "flow_matching.sparse_flow_matching",
    "ImageConditionedSparseFlowMatchingCFGTrainer": "flow_matching.sparse_flow_matching",
  
    "DinoToImageTrainerV1":"dino_to_image_trainer_v1",
    "RlaAutoencoderTrainer": "rla_autoencoder_trainer",
    "RLAWMTrainer": "rla_wm_trainer",
}

__submodules = []

__all__ = list(__attributes.keys()) + __submodules


def __getattr__(name):
    if name not in globals():
        if name in __attributes:
            module_name = __attributes[name]
            module = importlib.import_module(f".{module_name}", __name__)
            globals()[name] = getattr(module, name)
        elif name in __submodules:
            module = importlib.import_module(f".{name}", __name__)
            globals()[name] = module
        else:
            raise AttributeError(f"module {__name__} has no attribute {name}")
    return globals()[name]


# For Pylance
if __name__ == "__main__":
    from .basic import BasicTrainer as BasicTrainer

    from .vae.sparse_structure_vae import (
        SparseStructureVaeTrainer as SparseStructureVaeTrainer,
    )

    from .vae.structured_latent_vae_gaussian import (
        SLatVaeGaussianTrainer as SLatVaeGaussianTrainer,
    )
    from .vae.structured_latent_vae_rf_dec import (
        SLatVaeRadianceFieldDecoderTrainer as SLatVaeRadianceFieldDecoderTrainer,
    )
    from .vae.structured_latent_vae_mesh_dec import (
        SLatVaeMeshDecoderTrainer as SLatVaeMeshDecoderTrainer,
    )

    from .flow_matching.flow_matching import (
        FlowMatchingTrainer as FlowMatchingTrainer,
        FlowMatchingCFGTrainer as FlowMatchingCFGTrainer,
        TextConditionedFlowMatchingCFGTrainer as TextConditionedFlowMatchingCFGTrainer,
        ImageConditionedFlowMatchingCFGTrainer as ImageConditionedFlowMatchingCFGTrainer,
    )

    from .flow_matching.sparse_flow_matching import (
        SparseFlowMatchingTrainer as SparseFlowMatchingTrainer,
        SparseFlowMatchingCFGTrainer as SparseFlowMatchingCFGTrainer,
        TextConditionedSparseFlowMatchingCFGTrainer as TextConditionedSparseFlowMatchingCFGTrainer,
        ImageConditionedSparseFlowMatchingCFGTrainer as ImageConditionedSparseFlowMatchingCFGTrainer,
    )

    from .flow_matching.sparse_flow_matching_vcw import (
        VcwSparseFlowMatchingTrainer as VcwSparseFlowMatchingTrainer,
    )


    from .dino_dynamics_trainer_code_distill import DinoDynamicsTrainerVQDistill