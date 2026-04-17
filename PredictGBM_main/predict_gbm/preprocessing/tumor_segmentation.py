import time
import os
import shutil
import tempfile
import torch
import numpy as np
import nibabel as nib
from pathlib import Path
from loguru import logger
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

from PredictGBM_main.predict_gbm.utils.constants import (
    TUMORSEG_EDEMA_SCHEMA,
    TUMORSEG_SCHEMA,
    TUMORSEG_CORE_SCHEMA,
)


def split_segmentation(
    tumor_seg_file: Path,
    outdir: Path,
    necrotic_label: int = 1,
    edema_label: int = 2,
    enhancing_label: int = 3,
) -> None:
    """
    Split a composite tumor segmentation into separate binary segmentation files
    for enhancing/non-enhancing tumor and peritumoral edema.

    Parameters:
        tumor_seg_file (Path): Path to the input tumor segmentation NIfTI file.
        outdir (Path): Path to the output directory. Usually exam directory.
        necrotic_label (int): Label for necrotic / non-enhancing tissue in the segmentation.
        edema_label (int): Label for edema in the segmentation.
        enhancing_label (int): Label for enhancing tumor in the segmentation.

    Returns:
        None
    """
    logger.debug("Splitting tumor segmentation into core and edema.")
    tumor_seg = nib.load(str(tumor_seg_file))
    seg_data = np.rint(tumor_seg.get_fdata()).astype(np.int32)

    # Create a binary mask for non-enhancing and enhancing tumor (labels 1 and 3).
    enhancing_non_enhancing = nib.Nifti1Image(
        np.rint((seg_data == necrotic_label) | (seg_data == enhancing_label)).astype(
            np.int32
        ),
        affine=np.eye(4),
    )

    # Create a binary mask for edema (label 2).
    edema = nib.Nifti1Image(
        np.rint(seg_data == edema_label).astype(np.int32), affine=np.eye(4)
    )

    nib.save(enhancing_non_enhancing, str(TUMORSEG_CORE_SCHEMA.format(base_dir=outdir)))
    nib.save(edema, str(TUMORSEG_EDEMA_SCHEMA.format(base_dir=outdir)))
    logger.debug(
        f"Finished splitting segmentation. Output saved to {TUMORSEG_CORE_SCHEMA.format(base_dir=outdir).parent}."
    )





def run_brats(
    t1_file: Path,
    t1ce_file: Path,
    t2_file: Path,
    flair_file: Path,
    outdir: Path,
    pre_treatment: bool = True,
    cuda_device: str = "0",
) -> None:
    """
    使用 nnUNetv2 替换了原本的 brats 分割库。
    """
    start_time = time.time()
    logger.info("Starting tumor segmentation via nnUNetv2 (Replaced BRATS).")


    model_folder = r"D:\Our\Dataset002_BRATS19\nnUNetTrainer__nnUNetPlans__3d_fullres"

    device = torch.device(f"cuda:{cuda_device}" if torch.cuda.is_available() else "cpu")
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=device,
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True
    )
    predictor.initialize_from_trained_model_folder(
        model_folder, 
        use_folds=(0,), 
        checkpoint_name='checkpoint_final.pth'
    )

    with tempfile.TemporaryDirectory() as tmp_in, tempfile.TemporaryDirectory() as tmp_out:
       
        shutil.copy(str(t1_file), os.path.join(tmp_in, "tempcase_0000.nii.gz"))
        shutil.copy(str(t1ce_file),    os.path.join(tmp_in, "tempcase_0001.nii.gz"))
        shutil.copy(str(t2_file),  os.path.join(tmp_in, "tempcase_0002.nii.gz"))
        shutil.copy(str(flair_file),    os.path.join(tmp_in, "tempcase_0003.nii.gz"))
        
        predictor.predict_from_files(
            tmp_in,
            tmp_out,
            save_probabilities=False,
            overwrite=True,
            num_processes_preprocessing=1,
            num_processes_segmentation_export=1,
            folder_with_segs_from_prev_stage=None,
            num_parts=1,
            part_id=0
        )
        seg_outfile = str(TUMORSEG_SCHEMA.format(base_dir=outdir))
        os.makedirs(os.path.dirname(seg_outfile), exist_ok=True)
        predicted_file = os.path.join(tmp_out, "tempcase.nii.gz")
        if not os.path.exists(predicted_file):
            raise FileNotFoundError(f"nnUNet 预测失败，未在临时目录中找到 {predicted_file}。")
        shutil.move(predicted_file, seg_outfile)
    
    split_segmentation(Path(seg_outfile), outdir)

    time_spent = time.time() - start_time
    logger.info(
        f"Finished tumor segmentation via nnUNetv2 in {time_spent:.2f} seconds. Saved output to {seg_outfile}."
    )
    
