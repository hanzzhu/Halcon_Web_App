import os

import halcon as ha

Chadle_ProjectsDir = 'C:/Chadle_Projects'

Chadle_DataDir = Chadle_ProjectsDir + '/Chadle_Data'
Chadle_Halcon_ScriptsDir_OD = Chadle_ProjectsDir + '/Chadle_Halcon_Scripts/OD'
Halcon_DL_library_filesDir = Chadle_ProjectsDir + '/Halcon_DL_library_files'

ODProjectList = next(os.walk(Chadle_DataDir + '/Object_Detection'))[1]

TrainInfoDir = Chadle_Halcon_ScriptsDir_OD + '/TrainInfo.hdict'
EvaluationInfoDir = Chadle_Halcon_ScriptsDir_OD + '/EvaluationInfo.hdict'
PretrainedCompactModel_OD = Chadle_Halcon_ScriptsDir_OD + '/pretrained_dl_classifier_compact.hdl'
PretrainedEnhancedModel_OD = Chadle_Halcon_ScriptsDir_OD + '/pretrained_dl_classifier_enhanced.hdl'
###### Input/output directories ######
SeedRand = 42

# Root dir for OD
ObjectDetectionRootDir = Chadle_ProjectsDir+'/Chadle_Data/Object_Detection'

# Project dir, change to suit other projects
ProjectDir = ObjectDetectionRootDir + '/NTBW_Image Analytics'

# Pre_processDir: store pre-process data
Pre_processDir = ProjectDir + '/training_findinghub'

# ModelDir: store best and final models.
ModelDir = ProjectDir + '/models_hub'

# Path to the image directory.
HalconImageDir = ProjectDir + '/TapeRouting_Pictures'

# Create pretrained model for OD and pre-process sample folder
# ModelFileName = Pre_processDir + '/pretrained_dl_model_detection.hdl'

# Produce pre-process data in the pre-process folder.

# Create models during and after training.

# Label. TBD
Label_data = ObjectDetectionRootDir+'/NTBW_Image Analytics/NTBW_Initial_2.hdict'
###### Advanced Parameter settings ######


def setup_hdev_engine_OD():
    """Setup HDevEngine by setting procedure search paths."""

    engine = ha.HDevEngine()
    engine.set_procedure_path('C:/Program Files/MVTec/HALCON-20.11-Progress/procedures')

    # path where dl_training_PK.hdl and dl_visualization_PK.hdl files are located
    engine.set_procedure_path(Halcon_DL_library_filesDir)
    program = ha.HDevProgram(Chadle_Halcon_ScriptsDir_OD + '/DL_Train_OD_seagate.hdev')
    aug_call = ha.HDevProcedureCall(ha.HDevProcedure.load_local(program, 'augment_prepare'))
    preprocess_call = ha.HDevProcedureCall(ha.HDevProcedure.load_external('prepare_for_training_OD'))

    training_call = ha.HDevProcedureCall(ha.HDevProcedure.load_local(program, 'train_dl_model_CE'))
    # engine.set_procedure_path('E:/Customer evaluation/Seagate/HDev Engine_Python/dl_training_PK.hdpl')
    # engine.set_procedure_path('E:/Customer evaluation/Seagate/HDev Engine_Python/dl_visualization_PK.hdpl')
    # return aug_call, preprocess_call, training_call, evaluation_call


def estimate_values_OD(ImageWidth, ImageHeight, TrainingPercent, ValidationPercent, Label_data, ):
    setup_hdev_engine_OD()
    ha.set_system('seed_rand', SeedRand)

    # Create the output directory if it does not exist yet
    FileExists = ha.file_exists(Pre_processDir)
    if not FileExists:
        ha.make_dir(Pre_processDir)

    DLDataset_preprocess = ha.read_dict(Label_data, [], [])
    ha.remove_dict_key(DLDataset_preprocess, 'image_dir')
    ha.set_dict_tuple(DLDataset_preprocess, 'image_dir', HalconImageDir)

    proc_split = ha.HDevProcedure.load_external('split_dl_dataset')
    proc_call_split = ha.HDevProcedureCall(proc_split)

    proc_call_split.set_input_control_param_by_name('DLDataset', DLDataset_preprocess)
    proc_call_split.set_input_control_param_by_name('TrainingPercent', int(TrainingPercent))
    proc_call_split.set_input_control_param_by_name('ValidationPercent', int(ValidationPercent))
    proc_call_split.set_input_control_param_by_name('GenParam', [])
    proc_call_split.execute()

    GenParam = ha.create_dict()
    ha.set_dict_tuple(GenParam, 'split', 'train')

    proc_det_dl_param = ha.HDevProcedure.load_external('determine_dl_model_detection_param')
    proc_call_det_dl_param = ha.HDevProcedureCall(proc_det_dl_param)
    proc_call_det_dl_param.set_input_control_param_by_name('DLDataset', DLDataset_preprocess)
    proc_call_det_dl_param.set_input_control_param_by_name('ImageWidthTarget', int(ImageWidth))
    proc_call_det_dl_param.set_input_control_param_by_name('ImageHeightTarget', int(ImageHeight))
    proc_call_det_dl_param.set_input_control_param_by_name('GenParam', GenParam)

    proc_call_det_dl_param.execute()

    DLDetectionModelParam = proc_call_det_dl_param.get_output_control_param_by_name('DLDetectionModelParam')

    MinLevel = ha.get_dict_tuple(DLDetectionModelParam, 'min_level')
    MaxLevel = ha.get_dict_tuple(DLDetectionModelParam, 'max_level')
    AnchorNumSubscales = ha.get_dict_tuple(DLDetectionModelParam, 'anchor_num_subscales')
    AnchorAspectRatios = ha.get_dict_tuple(DLDetectionModelParam, 'anchor_aspect_ratios')

    print('MinLevel')
    print(MinLevel)
    print('MaxLevel')
    print(MaxLevel)
    print('AnchorNumSubscales')
    print(AnchorNumSubscales)
    print('AnchorAspectRatios')
    print(AnchorAspectRatios)

    return DLDataset_preprocess, MinLevel, MaxLevel, AnchorNumSubscales, AnchorAspectRatios


def preprocess_OD(ImWidth, ImHeight, ImageNumChannels, TrainingPercent, ValidationPercent, Label_data,
                  Backbone,
                  InstanceType, DLDataset_preprocess, MinLevel, MaxLevel,
                  AnchorNumSubscales, AnchorAspectRatios, NumClasses, Capacity):
    FileHandle = ha.open_file('mutex.dat', 'output')
    ha.fwrite_string(FileHandle, 0)
    if os.path.exists(TrainInfoDir):
        os.remove(TrainInfoDir)
    if os.path.exists(EvaluationInfoDir):
        os.remove(EvaluationInfoDir)
    if os.path.exists(PretrainedCompactModel_OD):
        os.remove(PretrainedCompactModel_OD)
    if os.path.exists(PretrainedEnhancedModel_OD):
        os.remove(PretrainedEnhancedModel_OD)

    ModelFileName = 'pretrained_dl_' + Backbone + '.hdl'
    # ModelFileName = Chadle_Halcon_ScriptsDir + '/pretrained_OD/pretrained_dl_classifier_enhanced.hdl'
    # Create the object detection model
    DLModelDetectionParam = ha.create_dict()

    ha.set_dict_tuple(DLModelDetectionParam, 'instance_type', 'rectangle1')
    ha.set_dict_tuple(DLModelDetectionParam, 'image_width', int(ImWidth))
    ha.set_dict_tuple(DLModelDetectionParam, 'image_height', int(ImHeight))
    ha.set_dict_tuple(DLModelDetectionParam, 'image_num_channels', int(ImageNumChannels))
    ha.set_dict_tuple(DLModelDetectionParam, 'min_level', MinLevel)
    ha.set_dict_tuple(DLModelDetectionParam, 'max_level', MaxLevel)
    ha.set_dict_tuple(DLModelDetectionParam, 'anchor_num_subscales', AnchorNumSubscales)
    ha.set_dict_tuple(DLModelDetectionParam, 'anchor_aspect_ratios', AnchorAspectRatios)
    ha.set_dict_tuple(DLModelDetectionParam, 'capacity', 'medium')
    ha.set_dict_tuple(DLModelDetectionParam, 'max_overlap', 0.4)
    ClassIDs = ha.get_dict_tuple(DLDataset_preprocess, 'class_ids')
    ha.set_dict_tuple(DLModelDetectionParam, 'class_ids', ClassIDs)

    proc_create_dl_model_detection = ha.HDevProcedure.load_external('create_dl_model_detection')
    proc_call_create_dl_model_detection = ha.HDevProcedureCall(proc_create_dl_model_detection)
    proc_call_create_dl_model_detection.set_input_control_param_by_name('Backbone', ModelFileName)
    proc_call_create_dl_model_detection.set_input_control_param_by_name('NumClasses', int(NumClasses))
    proc_call_create_dl_model_detection.set_input_control_param_by_name('DLModelDetectionParam',
                                                                        DLModelDetectionParam)

    proc_call_create_dl_model_detection.execute()
    DLModelHandle = proc_call_create_dl_model_detection.get_output_control_param_by_name('DLModelHandle')

    ha.write_dl_model(DLModelHandle, ModelFileName)

    proc_prep_param = ha.HDevProcedure.load_external('create_dl_preprocess_param_from_model')
    proc_call_prep_param = ha.HDevProcedureCall(proc_prep_param)

    proc_call_prep_param.set_input_control_param_by_name('DLModelHandle', DLModelHandle)
    proc_call_prep_param.set_input_control_param_by_name('NormalizationType', 'none')
    proc_call_prep_param.set_input_control_param_by_name('DomainHandling', 'full_domain')
    proc_call_prep_param.set_input_control_param_by_name('SetBackgroundID', [])
    proc_call_prep_param.set_input_control_param_by_name('ClassIDsBackground', [])
    proc_call_prep_param.set_input_control_param_by_name('GenParam', [])

    proc_call_prep_param.execute()
    DLPreprocessParam = proc_call_prep_param.get_output_control_param_by_name('DLPreprocessParam')
    ha.set_dict_tuple(DLPreprocessParam, 'instance_type', 'rectangle1')

    GenParam = ha.create_dict()
    ha.set_dict_tuple(GenParam, 'overwrite_files', 1)

    proc_preprocess_dl_dataset = ha.HDevProcedure.load_external('preprocess_dl_dataset')
    proc_call_preprocess_dl_dataset = ha.HDevProcedureCall(proc_preprocess_dl_dataset)
    Pre_processDataDirectory = Pre_processDir + '/dldataset_fipg_' + str(ImWidth) + 'x' + str(ImHeight)
    # Produce pre-process data in the pre-process folder.
    DLDatasetFileName = Pre_processDataDirectory + '/dl_dataset.hdict'
    DLPreprocessParamFileName = Pre_processDataDirectory + '/dl_preprocess_param.hdict'

    proc_call_preprocess_dl_dataset.set_input_control_param_by_name('DLDataset', DLDataset_preprocess)
    proc_call_preprocess_dl_dataset.set_input_control_param_by_name('DataDirectory', Pre_processDataDirectory)
    proc_call_preprocess_dl_dataset.set_input_control_param_by_name('DLPreprocessParam', DLPreprocessParam)
    proc_call_preprocess_dl_dataset.set_input_control_param_by_name('GenParam', GenParam)

    proc_call_preprocess_dl_dataset.execute()
    DLDatasetFileName = proc_call_preprocess_dl_dataset.get_output_control_param_by_name('DLDatasetFileName')

    ha.write_dict(DLPreprocessParam, DLPreprocessParamFileName, [], [])

    return DLDatasetFileName, DLPreprocessParamFileName, ModelFileName


def prepare_for_training_OD(AugmentationPercentage, Rotation, Mirror, BrightnessVariation, BrightnessVariationSpot,
                            RotationRange, BatchSize, InitialLearningRate, Momentum, NumEpochs,
                            ChangeLearningRateEpochs,
                            lr_change, WeightPrior, Class_Penalty, DLDatasetFileName, DLPreprocessParamFileName,
                            ModelFileName):
    BestModelBaseName = ModelDir + '/best_dl_model_detection'
    FinalModelBaseName = ModelDir + '/final_dl_model_detection'

    aug_call = ha.HDevProcedureCall(ha.HDevProcedure.load_external('augment_prepare'))

    aug_call.set_input_control_param_by_name('AugmentationPercentage', int(AugmentationPercentage))
    aug_call.set_input_control_param_by_name('Rotation', int(Rotation))
    aug_call.set_input_control_param_by_name('Mirror', str(Mirror))
    aug_call.set_input_control_param_by_name('BrightnessVariation', int(BrightnessVariation))
    aug_call.set_input_control_param_by_name('BrightnessVariationSpot', 0)
    aug_call.set_input_control_param_by_name('CropPercentage', 'off')
    aug_call.set_input_control_param_by_name('CropPixel', 'off')
    aug_call.set_input_control_param_by_name('RotationRange', 0)
    aug_call.set_input_control_param_by_name('IgnoreDirection', 'false')
    aug_call.set_input_control_param_by_name('ClassIDsNoOrientationExist', 'false')
    aug_call.set_input_control_param_by_name('ClassIDsNoOrientation', [])

    aug_call.execute()

    GenParamName_augment = aug_call.get_output_control_param_by_name('GenParamName_augment')
    GenParamValue_augment = aug_call.get_output_control_param_by_name('GenParamValue_augment')

    prepare_for_training_call = ha.HDevProcedureCall(ha.HDevProcedure.load_external('prepare_for_training_OD'))

    prepare_for_training_call.set_input_control_param_by_name('ExampleDataDir', Pre_processDir)
    prepare_for_training_call.set_input_control_param_by_name('ModelFileName', ModelFileName)
    # prepare_for_training_call.set_input_control_param_by_name('DataDirectory', DataDirectory)
    prepare_for_training_call.set_input_control_param_by_name('DLDatasetFileName', DLDatasetFileName)
    prepare_for_training_call.set_input_control_param_by_name('DLPreprocessParamFileName', DLPreprocessParamFileName)
    prepare_for_training_call.set_input_control_param_by_name('BestModelBaseName', BestModelBaseName)
    prepare_for_training_call.set_input_control_param_by_name('FinalModelBaseName', FinalModelBaseName)
    prepare_for_training_call.set_input_control_param_by_name('BatchSize', int(BatchSize))
    prepare_for_training_call.set_input_control_param_by_name('InitialLearningRate', float(InitialLearningRate))
    prepare_for_training_call.set_input_control_param_by_name('Momentum', float(Momentum))
    prepare_for_training_call.set_input_control_param_by_name('NumEpochs', int(NumEpochs))
    prepare_for_training_call.set_input_control_param_by_name('EvaluationIntervalEpochs', 1)

    ChangeLearningRateEpochsList = ChangeLearningRateEpochs.split(',')
    ChangeLearningRateEpochs = [int(i) for i in ChangeLearningRateEpochsList]
    prepare_for_training_call.set_input_control_param_by_name('ChangeLearningRateEpochs', ChangeLearningRateEpochs)

    lr_changeList = lr_change.split(',')
    lr_change = [float(i) for i in lr_changeList]
    prepare_for_training_call.set_input_control_param_by_name('lr_change', lr_change)

    prepare_for_training_call.set_input_control_param_by_name('WeightPrior', float(WeightPrior))
    prepare_for_training_call.set_input_control_param_by_name('GenParamName_augment', GenParamName_augment)
    prepare_for_training_call.set_input_control_param_by_name('GenParamValue_augment', GenParamValue_augment)

    Class_penaltyList = Class_Penalty.split(',')
    Class_Penalty = [float(i) for i in Class_penaltyList]
    prepare_for_training_call.set_input_control_param_by_name('Class_Penalty', Class_Penalty)

    prepare_for_training_call.execute()

    DLModelHandle = prepare_for_training_call.get_output_control_param_by_name('DLModelHandle')
    DLDataset = prepare_for_training_call.get_output_control_param_by_name('DLDataset')
    TrainParam = prepare_for_training_call.get_output_control_param_by_name('TrainParam')

    return DLModelHandle, DLDataset, TrainParam


def training_OD(DLDataset, DLModelHandle, TrainParam):
    proc_training = ha.HDevProcedure.load_external('train_dl_model_CE')
    training_call = ha.HDevProcedureCall(ha.HDevProcedure.load_external('train_dl_model_CE'))

    training_call.set_input_control_param_by_name('DLModelHandle', DLModelHandle)
    training_call.set_input_control_param_by_name('DLDataset', DLDataset)
    training_call.set_input_control_param_by_name('TrainParam', TrainParam)
    training_call.set_input_control_param_by_name('StartEpoch', 0)
    training_call.set_input_control_param_by_name('Display_Ctrl', 0)

    training_call.execute()


def get_TrainInfo_OD():
    if os.path.isfile(TrainInfoDir):
        try:
            TrainInfo = ha.read_dict(TrainInfoDir, (), ())
            time_elapsed = ha.get_dict_tuple(TrainInfo, 'time_elapsed')
            time_elapsed = time_elapsed[0]
            time_remaining = ha.get_dict_tuple(TrainInfo, 'time_remaining')
            time_remaining = time_remaining[0]
            epoch_traininfo = ha.get_dict_tuple(TrainInfo, 'epoch')
            epoch_traininfo = epoch_traininfo[0]
            loss_tuple = ha.get_dict_tuple(TrainInfo, 'mean_loss')
            loss_tuple = loss_tuple[0]
            num_iterations_per_epoch = ha.get_dict_tuple(TrainInfo, 'num_iterations_per_epoch')
            num_iterations_per_epoch = num_iterations_per_epoch[0]
            iteration = num_iterations_per_epoch * epoch_traininfo

        except:
            TrainInfo = ha.read_dict(TrainInfoDir, (), ())
            time_elapsed = ha.get_dict_tuple(TrainInfo, 'time_elapsed')
            time_elapsed = time_elapsed[0]
            time_remaining = ha.get_dict_tuple(TrainInfo, 'time_remaining')
            time_remaining = time_remaining[0]
            epoch_traininfo = ha.get_dict_tuple(TrainInfo, 'epoch')
            epoch_traininfo = epoch_traininfo[0]
            loss_tuple = ha.get_dict_tuple(TrainInfo, 'mean_loss')
            loss_tuple = loss_tuple[0]
            num_iterations_per_epoch = ha.get_dict_tuple(TrainInfo, 'num_iterations_per_epoch')
            num_iterations_per_epoch = num_iterations_per_epoch[0]
            iteration = num_iterations_per_epoch * epoch_traininfo
        return time_elapsed, time_remaining, epoch_traininfo, loss_tuple, iteration
    else:
        return False


def get_EvaluationInfo_OD():
    if os.path.isfile(EvaluationInfoDir):
        try:
            Evaluation_Info = ha.read_dict(EvaluationInfoDir, (), ())

            epoch_evaluation = ha.get_dict_tuple(Evaluation_Info, 'epoch')
            epoch_evaluation_value = epoch_evaluation[0]

            TrainSet_result = ha.get_dict_tuple(Evaluation_Info, 'result_train')
            TrainSet_result_max_num_detections_all = ha.get_dict_tuple(TrainSet_result, 'max_num_detections_all')
            TrainSet_area_all = ha.get_dict_tuple(TrainSet_result_max_num_detections_all, 'area_all')
            TrainSet_mean_ap =  ha.get_dict_tuple(TrainSet_area_all, 'mean_ap')

            ValidationSet_result = ha.get_dict_tuple(Evaluation_Info, 'result')
            ValidationSet_result_max_num_detections_all = ha.get_dict_tuple(ValidationSet_result, 'max_num_detections_all')
            ValidationSet_area_all = ha.get_dict_tuple(ValidationSet_result_max_num_detections_all, 'area_all')
            ValidationSet_mean_ap = ha.get_dict_tuple(ValidationSet_area_all, 'mean_ap')

            TrainSet_mean_ap_value = TrainSet_mean_ap[0]
            ValidationSet_mean_ap_value = ValidationSet_mean_ap[0]

        except:
            Evaluation_Info = ha.read_dict(EvaluationInfoDir, (), ())

            epoch_evaluation = ha.get_dict_tuple(Evaluation_Info, 'epoch')
            epoch_evaluation_value = epoch_evaluation[0]

            TrainSet_result = ha.get_dict_tuple(Evaluation_Info, 'result_train')
            TrainSet_result_max_num_detections_all = ha.get_dict_tuple(TrainSet_result, 'max_num_detections_all')
            TrainSet_area_all = ha.get_dict_tuple(TrainSet_result_max_num_detections_all, 'area_all')
            TrainSet_mean_ap = ha.get_dict_tuple(TrainSet_area_all, 'mean_ap')

            ValidationSet_result = ha.get_dict_tuple(Evaluation_Info, 'result')
            ValidationSet_result_max_num_detections_all = ha.get_dict_tuple(ValidationSet_result,
                                                                            'max_num_detections_all')
            ValidationSet_area_all = ha.get_dict_tuple(ValidationSet_result_max_num_detections_all, 'area_all')
            ValidationSet_mean_ap = ha.get_dict_tuple(ValidationSet_area_all, 'mean_ap')

            TrainSet_mean_ap_value = TrainSet_mean_ap[0]
            ValidationSet_mean_ap_value = ValidationSet_mean_ap[0]
        return epoch_evaluation_value, TrainSet_mean_ap_value, ValidationSet_mean_ap_value
    else:
        return False
