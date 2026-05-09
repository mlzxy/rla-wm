import torch

def get_background_ids(env):
    seg_id2name = {k: v.name for k, v in env.unwrapped.segmentation_id_map.items()}
    background_ids = [0]
    for seg_id, seg_name in seg_id2name.items():
        if seg_name == "ground":
            background_ids.append(seg_id)
    background_ids = torch.as_tensor(background_ids)
    return background_ids


def extract_rgbs_from_obs(obs, background_ids, cameras):
    """ return (1, num_cams, 3, H, W) tensor of RGB images with background masked out"""
    return torch.cat([
                (obs['sensor_data'][cam]['rgb'].float().permute(0, 3, 1, 2)
                 * (~torch.isin(obs['sensor_data'][cam]['segmentation'][0, :, :, 0], background_ids))[:, :]) / 255.
                for cam in cameras
            ])[None] 