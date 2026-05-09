import albumentations as A
import typing


def build_augmentation_pipeline(aug_config: dict) -> typing.Optional[A.ReplayCompose]:
    """
    Builds an augmentation pipeline based on the provided configuration.

    Args:
        aug_config: Dictionary containing augmentation parameters.

    Returns:
        An albumentations.ReplayCompose object or None if no augmentations are enabled.
    """
    if not aug_config:
        return None

    transforms_list = []

    # 1. Color Jitter (Brightness, Contrast, Saturation, Hue)
    brightness = aug_config.get("brightness", 0.0)
    contrast = aug_config.get("contrast", 0.0)
    saturation = aug_config.get("saturation", 0.0)
    hue = aug_config.get("hue", 0.0)
    p_color = aug_config.get("p_color", 1.0)

    if any([brightness, contrast, saturation, hue]):
        transforms_list.append(
            A.ColorJitter(
                brightness=brightness,
                contrast=contrast,
                saturation=saturation,
                hue=hue,
                p=p_color,
            )
        )

    # 2. Noise Augmentations
    # Gaussian Noise
    var_limit = aug_config.get("noise_var_limit", 0)
    p_noise = aug_config.get("p_noise", 0.0)
    if p_noise > 0 and var_limit != 0:
        # For albumentations 2.x, GaussNoise uses std_range which is sqrt of var
        if isinstance(var_limit, (tuple, list)):
            std_range = (var_limit[0] ** 0.5, var_limit[1] ** 0.5)
        else:
            std_limit = var_limit**0.5
            std_range = (0, std_limit)
        transforms_list.append(A.GaussNoise(std_range=std_range, p=p_noise))

    # ISO Noise
    p_iso = aug_config.get("p_iso", 0.0)
    if p_iso > 0:
        transforms_list.append(A.ISONoise(p=p_iso))

    # Multiplicative Noise
    p_mult_noise = aug_config.get("p_mult_noise", 0.0)
    if p_mult_noise > 0:
        transforms_list.append(A.MultiplicativeNoise(p=p_mult_noise))

    # 3. Blur Augmentations
    blur_limit = aug_config.get("blur_limit", 3)
    p_blur = aug_config.get("p_blur", 0.0)
    if p_blur > 0:
        transforms_list.append(A.GaussianBlur(blur_limit=blur_limit, p=p_blur))

    p_motion_blur = aug_config.get("p_motion_blur", 0.0)
    if p_motion_blur > 0:
        transforms_list.append(A.MotionBlur(blur_limit=blur_limit, p=p_motion_blur))

    # 4. Color Effects
    p_gray = aug_config.get("p_gray", 0.0)
    if p_gray > 0:
        transforms_list.append(A.ToGray(p=p_gray))

    p_posterize = aug_config.get("p_posterize", 0.0)
    if p_posterize > 0:
        transforms_list.append(A.Posterize(p=p_posterize))

    p_solarize = aug_config.get("p_solarize", 0.0)
    if p_solarize > 0:
        transforms_list.append(A.Solarize(p=p_solarize))

    # 5. Compression
    p_compress = aug_config.get("p_compress", 0.0)
    quality_lower = aug_config.get("compress_quality_lower", 50)
    quality_upper = aug_config.get("compress_quality_upper", 100)
    if p_compress > 0:
        transforms_list.append(
            A.ImageCompression(
                quality_range=(quality_lower, quality_upper), p=p_compress
            )
        )

    if transforms_list:
        print(f"Augmentation pipeline built with: {transforms_list}")
        return A.ReplayCompose(transforms_list)
    else:
        return None
