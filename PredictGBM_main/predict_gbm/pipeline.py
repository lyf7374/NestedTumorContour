import os
import time
from pathlib import Path
from loguru import logger
from typing import Dict, Optional
from PredictGBM_main.predict_gbm.utils.visualization import VisualizationPipe
from PredictGBM_main.predict_gbm.utils.constants import (
    PATIENT_PREOP_OUTPUT_SCHEMA,
    PATIENT_FOLLOWUP_OUTPUT_SCHEMA_1,
    PATIENT_FOLLOWUP_OUTPUT_SCHEMA_2,
)
from PredictGBM_main.predict_gbm.preprocessing.preprocess import (
    NiftiPreprocessor,
    RegisterRecurrencePipe,
)


class BaseProcessor:
    """
    Base class for Processors that perform pre-processing, prediction and evaluation.

    Parameters:
        patient_id (str): String acting as patient identifier. Included in the output directory structure.
        model_id (str): String identifying the growth model to be used.
        outdir (Path): Path to the output directory.
        cuda_device (Optional, str): The gpu device to use.
    """

    def __init__(
        self,
        patient_id: str,
        model_id: str,
        outdir: Path,
        cuda_device: str = "0",
    ) -> None:
        self.patient_id = patient_id
        self.model_id = model_id
        self.outdir = outdir
        self.cuda_device = cuda_device
        outdir_preop = PATIENT_PREOP_OUTPUT_SCHEMA.format(
            base_dir=self.outdir.absolute(), patient_id=self.patient_id
        )
        self.outdir_preop = outdir_preop
        

    def run(self) -> Dict[str, float]:
        os.environ["CUDA_VISIBLE_DEVICES"] = self.cuda_device
        start_time = time.time()
        self._preprocess_preop()


    def _preprocess_preop(self) -> None:  # pragma: no cover - interface
        raise NotImplementedError


class NiftiProcessor(BaseProcessor):
    """
    Performs preprocessing (skull stripping, normalization, atlas registration), growth model prediction
    and evaluation from a pre-operative MRI exam and a follow-up MRI exam passed as NIfTI images.
    Intermediate results such as tumor segmentation can be provided to skip the corresponding steps.

    Parameters:
        patient_id (str): String acting as patient identifier. Included in the output directory structure.
        model_id (str): String identifying the growth model to be used.
        t1_preop_file (Path): Path to the NIfTI file containing the pre-operative T1 image.
        t1c_preop_file (Path): Path to the NIfTI file containing the pre-operative T1c image.
        t2_preop_file (Path): Path to the NIfTI file containing the pre-operative T2 image.
        flair_preop_file (Path): Path to the NIfTI file containing the pre-operative FLAIR image.
        t1_followup_file (Path): Path to the NIfTI file containing the follow-up T1 image.
        t1c_followup_file (Path): Path to the NIfTI file containing the follow-up T1c image.
        t2_followup_file (Path): Path to the NIfTI file containing the follow-up T2 image.
        flair_followup_file (Path): Path to the NIfTI file containing the follow-up FLAIR image.
        outdir (Path): Path to the output directory.
        cuda_device (Optional, str): The gpu device to use.
        tumorseg_file (Optional, Path): Path to a NIfTI containing the pre-operative tumor segmentation
            that will be used instead of the BRATS algorithms. Expects labels 1 for necrotic, 2 for edema
            and 3 for enhancing tumor.
        recurrenceseg_file (Optional, Path): Path to a NIfTI containing the follow-up recurrence segmentation
            that will be used instead of the BRATS algorithms. Expects labels 1 for necrotic, 2 for edema
            and 3 for enhancing tumor.
        is_skull_stripped (Optional, bool): If true, skips the skull stripping step.
        is_coregistered (Optional, bool): If true, skips the co-registration to atlas space step.
            Note that BRATS algorithms were trained in SRI-24 space.
    """

    def __init__(
        self,
        patient_id: str,
        model_id: str,
        t1_preop_file: Path,
        t1c_preop_file: Path,
        t2_preop_file: Path,
        flair_preop_file: Path,
        outdir: Path,
        template_file : Path,
        tumorseg_file: Optional[Path] = None,
        is_coregistered: bool = False,
        is_skull_stripped: bool = False,
        cuda_device: str = "0",
    ) -> None:
        super().__init__(patient_id=patient_id,model_id=model_id,outdir=outdir,cuda_device=cuda_device)
        
        self.template_file = template_file
        self.t1_preop_file = t1_preop_file
        self.t1c_preop_file = t1c_preop_file
        self.t2_preop_file = t2_preop_file
        self.flair_preop_file = flair_preop_file

        

        
        self.tumorseg_file = tumorseg_file
        self.is_skull_stripped = is_skull_stripped
        self.is_coregistered = is_coregistered
        # TODO: This class can handle missing modalities IF the segmentations are provided.
        #      Currently, empty modalities can be handled as empty Path("") inputs.
        #      Implement this more explicitely and check if segmentations are provided properly.
        #      Might need to catch exceptions for visualization if modalities are missing.

#         时刻0
    def _preprocess_preop(self) -> None:
        preprocessor = NiftiPreprocessor(
            t1_file=self.t1_preop_file,
            t1ce_file=self.t1c_preop_file,
            t2_file=self.t2_preop_file,
            # adc_file = self.adc_preop_file,
            flair_file=self.flair_preop_file,
            template_file = self.template_file,
            outdir=self.outdir_preop,
            pre_treatment=True,
            cuda_device=self.cuda_device,
            is_coregistered=self.is_coregistered,
            is_skull_stripped=self.is_skull_stripped,
            tumorseg_file=self.tumorseg_file,
        )
        
        preprocessor.run()




