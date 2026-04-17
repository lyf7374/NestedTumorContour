import os
import ants
import time
import shutil
from pathlib import Path
from loguru import logger
from typing import Optional
from PredictGBM_main.predict_gbm.base import BasePipe
from PredictGBM_main.predict_gbm.preprocessing.tumor_segmentation import run_brats
from PredictGBM_main.predict_gbm.preprocessing.norm_ss_coregistration import (
    normalize,
    norm_ss_coregister,
    register_recurrence,
)
from PredictGBM_main.predict_gbm.preprocessing.tissue_segmentation import (
    generate_registration_mask,
    run_tissue_seg_registration,
)
from PredictGBM_main.predict_gbm.utils.utils import make_symlink
from PredictGBM_main.predict_gbm.utils.constants import (
    LONGITUDINAL_WARP_SCHEMA,
    MODALITY_CONVERTED_SCHEMA,
    MODALITY_STRIPPED_SCHEMA,
    RECURRENCE_SCHEMA,
    REGISTRATION_MASK_SCHEMA,
    REGISTRATION_TRAFO_SCHEMA,
    TUMOR_LABELS,
    TUMORSEG_SCHEMA,
)


class BasePreprocessor:
    """
    Base class for DicomPreprocessor and NiftiPreprocessor that handles shared pre-processing steps on a fixed directory
    structure.

    Parameters:
        outdir (Path): Output directory containing either nifti converted data or skull stripped niftis in the
            expected directory structure.
        pre_treatment (bool): Whether the provided DICOM are preop (True) or postop (False).
            Causes the BRATS segmentation algorithm to choose different models.
        mask_tissueseg (bool): If true, masks out the tumor region for the tissue segmentation step.
        perform_coregistration (bool): If true, performs normalization, skull stripping, co-registration
        perform_skull_stripping (bool): If true, performs skull stripping during co-registration step.
        perform_tumorseg (bool): If true, performs tumor segmentation via BRATS.
        perform_tissueseg (bool): If true, performs tissue segmentation.
        cuda_device (str): GPU device to use.
    """

    def __init__(
        self,
        outdir: Path,
        # pre_treatment: bool = False,
        pre_treatment: bool = True,

        mask_tissueseg: bool = True,
        perform_coregistration: bool = True,
        perform_skull_stripping: bool = False,
        # perform_tumorseg: bool = False,
        perform_tumorseg: bool = True,
        perform_tissueseg: bool = True,
        cuda_device: str = "0",
        template_file: Path=None,
        
    ) -> None:
    
        if template_file is None:
            self.template_file = Path("templates/T1_brain.nii")
        self.outdir = outdir
        self.pre_treatment = pre_treatment
        self.mask_tissueseg = True
        self.template_file = template_file # MODIFIED: Stored template_file
        self.perform_coregistration = perform_coregistration
        self.perform_skull_stripping = perform_skull_stripping
        # self.perform_tumorseg = False
        self.perform_tumorseg = True
        self.perform_tissueseg = True
        self.cuda_device = cuda_device

    def run(self) -> None:

        start_time = time.time()
        os.environ["CUDA_VISIBLE_DEVICES"] = self.cuda_device
        logger.info("Starting preprocessing.")

        
        
        print("---------------------------------------------------------------开始配准！！！")
        
        if self.perform_coregistration:
            print("--------------------------------------------------运行的是第一个perform_coregistration")
            norm_ss_coregister(
                t1_file=MODALITY_CONVERTED_SCHEMA.format(
                    base_dir=self.outdir, modality="t1"
                ),
                t1ce_file=MODALITY_CONVERTED_SCHEMA.format(
                    base_dir=self.outdir, modality="t1ce"
                ),
                t2_file=MODALITY_CONVERTED_SCHEMA.format(
                    base_dir=self.outdir, modality="t2"
                ),
                flair_file=MODALITY_CONVERTED_SCHEMA.format(
                    base_dir=self.outdir, modality="flair"
                ),
                skull_strip=self.perform_skull_stripping,
                outdir=self.outdir,
                template_file = self.template_file, # MODIFIED: Passed template_file to the function
            )
        print("---------------------------------------------------------------配准结束！！！")
# #   norunning 
        
        if self.perform_tumorseg:
            print("----------------------------------------------------------------运行的是tumorseg部分")
            run_brats(
                t1_file=MODALITY_STRIPPED_SCHEMA.format(
                    base_dir=self.outdir, modality="t1"
                ),
                t1ce_file=MODALITY_STRIPPED_SCHEMA.format(
                    base_dir=self.outdir, modality="t1ce"
                ),
                t2_file=MODALITY_STRIPPED_SCHEMA.format(
                    base_dir=self.outdir, modality="t2"
                ),
                flair_file=MODALITY_STRIPPED_SCHEMA.format(
                    base_dir=self.outdir, modality="flair"
                ),
                outdir=self.outdir,
                pre_treatment=self.pre_treatment,
                cuda_device=self.cuda_device,
            )
        tissueseg_kwargs = {}
        if self.mask_tissueseg:
            print("-------------------------------------------------------运行的是mask_tissueseg")
            
            tumor_mask_path = TUMORSEG_SCHEMA.format(base_dir=self.outdir)
            
            registration_mask_file = REGISTRATION_MASK_SCHEMA.format(base_dir=self.outdir)
            
            generate_registration_mask(
                tumor_seg_file=tumor_mask_path,
                outfile=registration_mask_file,
            )
            tissueseg_kwargs["registration_mask_file"] = str(registration_mask_file)
        if self.perform_tissueseg:
            
            print("-------------------------------------------------------运行的是perform_tissueseg")
            run_tissue_seg_registration(
                t1_file=MODALITY_STRIPPED_SCHEMA.format(
                    base_dir=self.outdir, modality="t1ce"
                ),
                outdir=self.outdir,
                **tissueseg_kwargs,
            )

        time_spent = time.time() - start_time
        logger.info(
            f"Finished preprocessing in {time_spent:.2f} seconds. Results saved to {self.outdir}."
        )
        


class NiftiPreprocessor(BasePreprocessor):
    """
    Performs a multitude of precessing steps to prepare nifti inputs for tumor growth models. Allows passing
    available intermediate results like tumor segmentation or already skull stripped images.

    Parameters:
        t1_file (Path): Path to the NIfTI file with the t1 data.
        t1c_file (Path): Path to the NIfTI file with the t1c data.
        t2_file (Path): Path to the NIfTI file with the t2 data.
        flair_file (Path): Path to the NIfTI file with the flair data.
        pre_treatment (bool): Whether the provided DICOM are preop (True) or postop (False).
            Causes the BRATS segmentation algorithm to choose different models.
        outdir (Path): Base directory for the output. Usually exam directory.
        cuda_device (str): GPU device to use.
        is_coregistered (bool): True if the provided data has already been co-registered to SRI-24 space and skull stripped.
        is_skull_stripped (bool): True if the provided data has already been normalized, skull stripped and co-registered.
        tumorseg_file (Optional, Path): Path to the tumor segmentation.
    """
    
    def __init__(
        self,
        t1_file: Path,
        t1ce_file: Path,
        t2_file: Path,
        flair_file: Path,
        outdir: Path,
        pre_treatment: bool,
        is_coregistered: bool,
        is_skull_stripped: bool,
        tumorseg_file: Optional[Path] = None,
        # adc_file: Optional[Path] = None, # 修改：将 adc_file 设为可选
        template_file: Path=None,
        cuda_device: str = "0",
    ) -> None:
        
    
        if template_file is None:
            self.template_file =Path("templates/T1_brain.nii")
        super().__init__(
            outdir=outdir,
            pre_treatment=pre_treatment,
            cuda_device=cuda_device,
            template_file=template_file, # MODIFIED: Passed template_file to super()
            # T,F
            perform_coregistration=is_coregistered,
            perform_skull_stripping=is_skull_stripped,
            perform_tumorseg=(tumorseg_file is None),
            perform_tissueseg=pre_treatment,
        )
        
        self.t1_file = t1_file
        self.t1ce_file = t1ce_file
        self.t2_file = t2_file
        self.flair_file = flair_file
        self.is_coregistered = is_coregistered
        self.is_skull_stripped = is_skull_stripped
        self.tumorseg_file = tumorseg_file

    def run(self) -> None:

        os.environ["CUDA_VISIBLE_DEVICES"] = self.cuda_device

        modality_dict = {
            "t1": self.t1_file,
            "t1ce": self.t1ce_file,
            "t2": self.t2_file,
            "flair": self.flair_file,
        }

        if not self.is_coregistered:
            logger.info("Running with provided skull stripped modality images.")
            for modality, path in modality_dict.items():
                if str(path) == ".":
                    continue  # ignore missing modalities
                normalize(
                    img_file=path,
                    outfile=MODALITY_STRIPPED_SCHEMA.format(
                        base_dir=self.outdir, modality=modality
                    ),
                )
        else:
            for modality, path in modality_dict.items():
                make_symlink(
                    src=path,
                    dst=MODALITY_CONVERTED_SCHEMA.format(
                        base_dir=self.outdir, modality=modality
                    ),
                )
                
        super().run()



class RegisterRecurrencePipe(BasePipe):
    """Performs longitudinal registration, transforming followup t1c and recurrence segmentation to preop space."""

    def __init__(
        self,
        preop_dir: Path,
        followup_dir: Path,
        is_coregistered: bool = False,
        use_fixed_mask: bool = False,
        use_moving_mask: bool = False,
    ) -> None:
        super().__init__(preop_dir=preop_dir, followup_dir=followup_dir)
        self.is_coregistered = is_coregistered
        self.use_fixed_mask = use_fixed_mask
        self.use_moving_mask = use_moving_mask

    def run(self) -> None:  # pragma: no cover - wrapper tested via pipeline
        start_time = time.time()
        logger.info("Starting longitudinal processing.")

        t1ce_pre_file = MODALITY_STRIPPED_SCHEMA.format(
            base_dir=self.preop_dir, modality="t1ce"
        )
        t1ce_post_file = MODALITY_STRIPPED_SCHEMA.format(
            base_dir=self.followup_dir, modality="t1ce"
        )
        recurrence_seg_file = TUMORSEG_SCHEMA.format(base_dir=self.followup_dir)

        reg_kwargs = {}
        if self.use_fixed_mask:
            reg_kwargs["fixed_mask_file"] = REGISTRATION_MASK_SCHEMA.format(
                base_dir=self.preop_dir
            )
        if self.use_moving_mask:
            reg_kwargs["moving_mask_file"] = REGISTRATION_MASK_SCHEMA.format(
                base_dir=self.followup_dir
            )
        
        if self.is_coregistered:
            make_symlink(
                src=t1ce_post_file,
                dst=LONGITUDINAL_WARP_SCHEMA.format(base_dir=self.followup_dir),
            )
            make_symlink(
                src=recurrence_seg_file,
                dst=RECURRENCE_SCHEMA.format(base_dir=self.followup_dir),
            )
        else:
            register_recurrence(
                t1ce_pre_file=t1ce_pre_file,
                t1ce_post_file=t1ce_post_file,
                recurrence_seg_file=recurrence_seg_file,
                outdir=self.followup_dir,
                **reg_kwargs,
            )
            
        time_spent = time.time() - start_time
        logger.info(
            f"Finished longitudinal preprocessing in {time_spent:.2f} seconds. Results saved to {self.followup_dir}."
        )
