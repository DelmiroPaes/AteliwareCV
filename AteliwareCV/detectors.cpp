//
// Copyright (C) 2019 Ateliware
//

/************************************************************************************************
 * \file  detectors.cpp.
 *
 * \brief Implements the detectors class
 **************************************************************************************************/

#pragma warning( disable : 4251 )
#pragma warning( disable : 4267 )
#pragma warning( disable : 4275 )

#include <gflags/gflags.h>
#include <functional>
#include <iostream>
#include <fstream>
#include <random>
#include <memory>
#include <chrono>
#include <vector>
#include <string>
#include <utility>
#include <algorithm>
#include <iterator>
#include <map>

#include <inference_engine.hpp>

#include <samples/ocv_common.hpp>
#include <samples/slog.hpp>

#include <ie_iextension.h>
#ifdef WITH_EXTENSIONS
#include <ext_list.hpp>
#endif

#include "detectors.hpp"

using namespace InferenceEngine;


/**********************************************************************************************//**
 * \fn  BaseDetection::BaseDetection(std::string topoName, const std::string &pathToModel, const std::string &deviceForInference, int maxBatch, bool isBatchDynamic, bool isAsync, bool doRawOutputMessages)
 *
 * \brief Constructor
 *
 * \author  Delmiro Paes
 *
 * \param topoName            Name of the topo.
 * \param pathToModel         The path to model.
 * \param deviceForInference  The device for inference.
 * \param maxBatch            The maximum batch.
 * \param isBatchDynamic      True if is batch dynamic, false if not.
 * \param isAsync             True if is asynchronous, false if not.
 * \param doRawOutputMessages True to do raw output messages.
 **************************************************************************************************/
BaseDetection::BaseDetection(std::string topoName,
  const std::string& pathToModel,
  const std::string& deviceForInference,
  int maxBatch, bool isBatchDynamic, bool isAsync,
  bool doRawOutputMessages)
  : topoName(topoName), pathToModel(pathToModel), deviceForInference(deviceForInference),
  maxBatch(maxBatch), isBatchDynamic(isBatchDynamic), isAsync(isAsync),
  enablingChecked(false), _enabled(false), doRawOutputMessages(doRawOutputMessages) 
{
  if (isAsync) 
  {
    slog::info << "Use async mode for " << topoName << slog::endl;
  }
}


/**********************************************************************************************//**
 * \fn  BaseDetection::~BaseDetection()
 *
 * \brief Destructor
 *
 * \author  Delmiro Paes
 **************************************************************************************************/
BaseDetection::~BaseDetection() {}


/**********************************************************************************************//**
 * \fn  ExecutableNetwork* BaseDetection::operator->()
 *
 * \brief Member dereference operator
 *
 * \author  Delmiro Paes
 *
 * \returns The dereferenced object.
 **************************************************************************************************/
ExecutableNetwork* BaseDetection::operator ->() 
{
  return &net;
}


/**********************************************************************************************//**
 * \fn  void BaseDetection::submitRequest()
 *
 * \brief Submit request
 *
 * \author  Delmiro Paes
 **************************************************************************************************/
void BaseDetection::submitRequest() 
{
  if (!enabled() || request == nullptr) return;
  if (isAsync) 
  {
    request->StartAsync();
  }
  else 
  {
    request->Infer();
  }
}


/**********************************************************************************************//**
 * \fn  void BaseDetection::wait()
 *
 * \brief Waits this object
 *
 * \author  Delmiro Paes
 **************************************************************************************************/
void BaseDetection::wait() 
{
  if (!enabled() || !request || !isAsync)
    return;
  request->Wait(IInferRequest::WaitMode::RESULT_READY);
}


/**********************************************************************************************//**
 * \fn  bool BaseDetection::enabled() const
 *
 * \brief Enabled this object
 *
 * \author  Delmiro Paes
 *
 * \returns True if it succeeds, false if it fails.
 **************************************************************************************************/
bool BaseDetection::enabled() const 
{
  if (!enablingChecked) 
  {
    _enabled = !pathToModel.empty();
    if (!_enabled) 
    {
      slog::info << topoName << " DISABLED" << slog::endl;
    }
    enablingChecked = true;
  }
  return _enabled;
}


/**********************************************************************************************//**
 * \fn  void BaseDetection::printPerformanceCounts(std::string fullDeviceName)
 *
 * \brief Print performance counts
 *
 * \author  Delmiro Paes
 *
 * \param fullDeviceName  Name of the full device.
 **************************************************************************************************/
void BaseDetection::printPerformanceCounts(std::string fullDeviceName) 
{
  if (!enabled()) 
  {
    return;
  }
  slog::info << "Performance counts for " << topoName << slog::endl << slog::endl;
  ::printPerformanceCounts(*request, std::cout, fullDeviceName, false);
}


/**********************************************************************************************//**
 * \fn  FaceDetection::FaceDetection(const std::string &pathToModel, const std::string &deviceForInference, int maxBatch, bool isBatchDynamic, bool isAsync, double detectionThreshold, bool doRawOutputMessages, float bb_enlarge_coefficient, float bb_dx_coefficient, float bb_dy_coefficient)
 *
 * \brief Constructor
 *
 * \author  Delmiro Paes
 *
 * \param pathToModel             The path to model.
 * \param deviceForInference      The device for inference.
 * \param maxBatch                The maximum batch.
 * \param isBatchDynamic          True if is batch dynamic, false if not.
 * \param isAsync                 True if is asynchronous, false if not.
 * \param detectionThreshold      The detection threshold.
 * \param doRawOutputMessages     True to do raw output messages.
 * \param bb_enlarge_coefficient  The bb enlarge coefficient.
 * \param bb_dx_coefficient       The bb dx coefficient.
 * \param bb_dy_coefficient       The bb dy coefficient.
 **************************************************************************************************/
FaceDetection::FaceDetection(const std::string& pathToModel,
  const std::string& deviceForInference,
  int maxBatch, bool isBatchDynamic, bool isAsync,
  double detectionThreshold, bool doRawOutputMessages,
  float bb_enlarge_coefficient, float bb_dx_coefficient, float bb_dy_coefficient)
  : BaseDetection("Face Detection", pathToModel, deviceForInference, maxBatch, isBatchDynamic, isAsync, doRawOutputMessages),
  detectionThreshold(detectionThreshold),
  maxProposalCount(0), objectSize(0), enquedFrames(0), width(0), height(0),
  bb_enlarge_coefficient(bb_enlarge_coefficient), bb_dx_coefficient(bb_dx_coefficient),
  bb_dy_coefficient(bb_dy_coefficient), resultsFetched(false) {}


/**********************************************************************************************//**
 * \fn  void FaceDetection::submitRequest()
 *
 * \brief Submit request
 *
 * \author  Delmiro Paes
 **************************************************************************************************/
void FaceDetection::submitRequest() 
{
  if (!enquedFrames) return;
  enquedFrames = 0;
  resultsFetched = false;
  results.clear();
  BaseDetection::submitRequest();
}


/**********************************************************************************************//**
 * \fn  void FaceDetection::enqueue(const cv::Mat &frame)
 *
 * \brief Adds an object onto the end of this queue
 *
 * \author  Delmiro Paes
 *
 * \param frame The frame.
 **************************************************************************************************/
void FaceDetection::enqueue(const cv::Mat& frame) 
{
  if (!enabled()) return;

  if (!request) 
  {
    request = net.CreateInferRequestPtr();
  }

  width = static_cast<float>(frame.cols);
  height = static_cast<float>(frame.rows);

  Blob::Ptr  inputBlob = request->GetBlob(input);

  matU8ToBlob<uint8_t>(frame, inputBlob);

  enquedFrames = 1;
}


/**********************************************************************************************//**
 * \fn  CNNNetwork FaceDetection::read()
 *
 * \brief Gets the read
 *
 * \author  Delmiro Paes
 *
 * \exception std::logic_error  Raised when a logic error condition occurs.
 *
 * \returns A CNNNetwork.
 **************************************************************************************************/
CNNNetwork FaceDetection::read() 
{
  slog::info << "Loading network files for Face Detection" << slog::endl;
  CNNNetReader netReader;
  /** Read network model **/
  netReader.ReadNetwork(pathToModel);
  /** Set batch size to 1 **/
  slog::info << "Batch size is set to " << maxBatch << slog::endl;
  netReader.getNetwork().setBatchSize(maxBatch);
  /** Extract model name and load its weights **/
  std::string binFileName = fileNameNoExt(pathToModel) + ".bin";
  netReader.ReadWeights(binFileName);
  /** Read labels (if any)**/
  std::string labelFileName = fileNameNoExt(pathToModel) + ".labels";

  std::ifstream inputFile(labelFileName);
  std::copy(std::istream_iterator<std::string>(inputFile),
    std::istream_iterator<std::string>(),
    std::back_inserter(labels));
  // -----------------------------------------------------------------------------------------------------

  /** SSD-based network should have one input and one output **/
  // ---------------------------Check inputs -------------------------------------------------------------
  slog::info << "Checking Face Detection network inputs" << slog::endl;
  InputsDataMap inputInfo(netReader.getNetwork().getInputsInfo());
  if (inputInfo.size() != 1) 
  {
    throw std::logic_error("Face Detection network should have only one input");
  }
  InputInfo::Ptr inputInfoFirst = inputInfo.begin()->second;
  inputInfoFirst->setPrecision(Precision::U8);
  // -----------------------------------------------------------------------------------------------------

  // ---------------------------Check outputs ------------------------------------------------------------
  slog::info << "Checking Face Detection network outputs" << slog::endl;
  OutputsDataMap outputInfo(netReader.getNetwork().getOutputsInfo());
  if (outputInfo.size() != 1) 
  {
    throw std::logic_error("Face Detection network should have only one output");
  }
  DataPtr& _output = outputInfo.begin()->second;
  output = outputInfo.begin()->first;

  const CNNLayerPtr outputLayer = netReader.getNetwork().getLayerByName(output.c_str());
  if (outputLayer->type != "DetectionOutput") 
  {
    throw std::logic_error("Face Detection network output layer(" + outputLayer->name +
      ") should be DetectionOutput, but was " + outputLayer->type);
  }

  if (outputLayer->params.find("num_classes") == outputLayer->params.end()) 
  {
    throw std::logic_error("Face Detection network output layer (" +
      output + ") should have num_classes integer attribute");
  }

  const size_t num_classes = outputLayer->GetParamAsUInt("num_classes");
  if (labels.size() != num_classes) 
  {
    if (labels.size() == (num_classes - 1))  // if network assumes default "background" class, which has no label
      labels.insert(labels.begin(), "fake");
    else
      labels.clear();
  }

  const SizeVector outputDims = _output->getTensorDesc().getDims();
  maxProposalCount = outputDims[2];
  objectSize = outputDims[3];

  if (objectSize != 7) 
  {
    throw std::logic_error("Face Detection network output layer should have 7 as a last dimension");
  }

  if (outputDims.size() != 4) 
  {
    throw std::logic_error("Face Detection network output dimensions not compatible shoulld be 4, but was " +
      std::to_string(outputDims.size()));
  }
  _output->setPrecision(Precision::FP32);

  slog::info << "Loading Face Detection model to the " << deviceForInference << " device" << slog::endl;
  input = inputInfo.begin()->first;
  return netReader.getNetwork();
}


/**********************************************************************************************//**
 * \fn  void FaceDetection::fetchResults()
 *
 * \brief Fetches the results
 *
 * \author  Delmiro Paes
 **************************************************************************************************/
void FaceDetection::fetchResults() 
{
  if (!enabled()) return;
  results.clear();
  if (resultsFetched) return;
  resultsFetched = true;
  const float* detections = request->GetBlob(output)->buffer().as<float*>();

  for (int i = 0; i < maxProposalCount; i++) 
  {
    float image_id = detections[i * objectSize + 0];
    if (image_id < 0) 
    {
      break;
    }
    Result r;
    r.label = static_cast<int>(detections[i * objectSize + 1]);
    r.confidence = detections[i * objectSize + 2];

    if (r.confidence <= detectionThreshold && !doRawOutputMessages) 
    {
      continue;
    }

    r.location.x = static_cast<int>(detections[i * objectSize + 3] * width);
    r.location.y = static_cast<int>(detections[i * objectSize + 4] * height);
    r.location.width = static_cast<int>(detections[i * objectSize + 5] * width - r.location.x);
    r.location.height = static_cast<int>(detections[i * objectSize + 6] * height - r.location.y);

    // Make square and enlarge face bounding box for more robust operation of face analytics networks
    int bb_width = r.location.width;
    int bb_height = r.location.height;

    int bb_center_x = r.location.x + bb_width / 2;
    int bb_center_y = r.location.y + bb_height / 2;

    int max_of_sizes = std::max(bb_width, bb_height);

    int bb_new_width = static_cast<int>(bb_enlarge_coefficient * max_of_sizes);
    int bb_new_height = static_cast<int>(bb_enlarge_coefficient * max_of_sizes);

    r.location.x = bb_center_x - static_cast<int>(std::floor(bb_dx_coefficient * bb_new_width / 2));
    r.location.y = bb_center_y - static_cast<int>(std::floor(bb_dy_coefficient * bb_new_height / 2));

    r.location.width = bb_new_width;
    r.location.height = bb_new_height;

    if (doRawOutputMessages) 
    {
      std::cout << "[" << i << "," << r.label << "] element, prob = " << r.confidence <<
        "    (" << r.location.x << "," << r.location.y << ")-(" << r.location.width << ","
        << r.location.height << ")"
        << ((r.confidence > detectionThreshold) ? " WILL BE RENDERED!" : "") << std::endl;
    }

    if (r.confidence > detectionThreshold) 
    {
      results.push_back(r);
    }
  }
}


/**********************************************************************************************//**
 * \fn  AgeGenderDetection::AgeGenderDetection(const std::string &pathToModel, const std::string &deviceForInference, int maxBatch, bool isBatchDynamic, bool isAsync, bool doRawOutputMessages)
 *
 * \brief Constructor
 *
 * \author  Delmiro Paes
 *
 * \param pathToModel         The path to model.
 * \param deviceForInference  The device for inference.
 * \param maxBatch            The maximum batch.
 * \param isBatchDynamic      True if is batch dynamic, false if not.
 * \param isAsync             True if is asynchronous, false if not.
 * \param doRawOutputMessages True to do raw output messages.
 **************************************************************************************************/
AgeGenderDetection::AgeGenderDetection(const std::string& pathToModel,
  const std::string& deviceForInference,
  int maxBatch, bool isBatchDynamic, bool isAsync, bool doRawOutputMessages)
  : BaseDetection("Age/Gender", pathToModel, deviceForInference, maxBatch, isBatchDynamic, isAsync, doRawOutputMessages),
  enquedFaces(0) 
{
}


/**********************************************************************************************//**
 * \fn  void AgeGenderDetection::submitRequest()
 *
 * \brief Submit request
 *
 * \author  Delmiro Paes
 **************************************************************************************************/
void AgeGenderDetection::submitRequest() 
{
  if (!enquedFaces)
    return;
  if (isBatchDynamic) {
    request->SetBatch(enquedFaces);
  }
  BaseDetection::submitRequest();
  enquedFaces = 0;
}


/**********************************************************************************************//**
 * \fn  void AgeGenderDetection::enqueue(const cv::Mat &face)
 *
 * \brief Adds an object onto the end of this queue
 *
 * \author  Delmiro Paes
 *
 * \param face  The face.
 **************************************************************************************************/
void AgeGenderDetection::enqueue(const cv::Mat& face) 
{
  if (!enabled()) 
  {
    return;
  }

  if (enquedFaces == maxBatch) 
  {
    slog::warn << "Number of detected faces more than maximum(" << maxBatch << ") processed by Age/Gender Recognition network" << slog::endl;
    return;
  }

  if (!request) 
  {
    request = net.CreateInferRequestPtr();
  }

  Blob::Ptr  inputBlob = request->GetBlob(input);

  matU8ToBlob<uint8_t>(face, inputBlob, enquedFaces);

  enquedFaces++;
}


/**********************************************************************************************//**
 * \fn  AgeGenderDetection::Result AgeGenderDetection::operator[] (int idx) const
 *
 * \brief Array indexer operator
 *
 * \author  Delmiro Paes
 *
 * \param idx Zero-based index of the.
 *
 * \returns The indexed value.
 **************************************************************************************************/
AgeGenderDetection::Result AgeGenderDetection::operator[] (int idx) const 
{
  Blob::Ptr  genderBlob = request->GetBlob(outputGender);
  Blob::Ptr  ageBlob = request->GetBlob(outputAge);

  AgeGenderDetection::Result r = { ageBlob->buffer().as<float*>()[idx] * 100, genderBlob->buffer().as<float*>()[idx * 2 + 1] };
  if (doRawOutputMessages) 
  {
    std::cout << "[" << idx << "] element, male prob = " << r.maleProb << ", age = " << r.age << std::endl;
  }

  return r;
}


/**********************************************************************************************//**
 * \fn  CNNNetwork AgeGenderDetection::read()
 *
 * \brief Gets the read
 *
 * \author  Delmiro Paes
 *
 * \exception std::logic_error  Raised when a logic error condition occurs.
 *
 * \returns A CNNNetwork.
 **************************************************************************************************/
CNNNetwork AgeGenderDetection::read() 
{
  slog::info << "Loading network files for Age/Gender Recognition network" << slog::endl;
  CNNNetReader netReader;
  // Read network
  netReader.ReadNetwork(pathToModel);

  // Set maximum batch size to be used.
  netReader.getNetwork().setBatchSize(maxBatch);
  slog::info << "Batch size is set to " << netReader.getNetwork().getBatchSize() << " for Age/Gender Recognition network" << slog::endl;


  // Extract model name and load its weights
  std::string binFileName = fileNameNoExt(pathToModel) + ".bin";
  netReader.ReadWeights(binFileName);

  // ---------------------------Check inputs -------------------------------------------------------------
  // Age/Gender Recognition network should have one input and two outputs
  slog::info << "Checking Age/Gender Recognition network inputs" << slog::endl;
  InputsDataMap inputInfo(netReader.getNetwork().getInputsInfo());
  if (inputInfo.size() != 1) 
  {
    throw std::logic_error("Age/Gender Recognition network should have only one input");
  }
  InputInfo::Ptr& inputInfoFirst = inputInfo.begin()->second;
  inputInfoFirst->setPrecision(Precision::U8);
  input = inputInfo.begin()->first;
  // -----------------------------------------------------------------------------------------------------

  // ---------------------------Check outputs ------------------------------------------------------------
  slog::info << "Checking Age/Gender Recognition network outputs" << slog::endl;
  OutputsDataMap outputInfo(netReader.getNetwork().getOutputsInfo());
  if (outputInfo.size() != 2) 
  {
    throw std::logic_error("Age/Gender Recognition network should have two output layers");
  }
  auto it = outputInfo.begin();

  DataPtr ptrAgeOutput = (it++)->second;
  DataPtr ptrGenderOutput = (it++)->second;

  if (!ptrAgeOutput) 
  {
    throw std::logic_error("Age output data pointer is not valid");
  }

  if (!ptrGenderOutput) 
  {
    throw std::logic_error("Gender output data pointer is not valid");
  }

  auto genderCreatorLayer = ptrGenderOutput->getCreatorLayer().lock();
  auto ageCreatorLayer = ptrAgeOutput->getCreatorLayer().lock();

  if (!ageCreatorLayer) 
  {
    throw std::logic_error("Age creator layer pointer is not valid");
  }

  if (!genderCreatorLayer) 
  {
    throw std::logic_error("Gender creator layer pointer is not valid");
  }

  // if gender output is convolution, it can be swapped with age
  if (genderCreatorLayer->type == "Convolution") 
  {
    std::swap(ptrAgeOutput, ptrGenderOutput);
  }

  if (ptrAgeOutput->getCreatorLayer().lock()->type != "Convolution") 
  {
    throw std::logic_error("In Age/Gender Recognition network, age layer (" + ageCreatorLayer->name +
      ") should be a Convolution, but was: " + ageCreatorLayer->type);
  }

  if (ptrGenderOutput->getCreatorLayer().lock()->type != "SoftMax") 
  {
    throw std::logic_error("In Age/Gender Recognition network, gender layer (" + genderCreatorLayer->name +
      ") should be a SoftMax, but was: " + genderCreatorLayer->type);
  }

  slog::info << "Age layer: " << ageCreatorLayer->name << slog::endl;
  slog::info << "Gender layer: " << genderCreatorLayer->name << slog::endl;

  outputAge = ptrAgeOutput->getName();
  outputGender = ptrGenderOutput->getName();

  slog::info << "Loading Age/Gender Recognition model to the " << deviceForInference << " plugin" << slog::endl;
  _enabled = true;

  return netReader.getNetwork();
}


/**********************************************************************************************//**
 * \fn  HeadPoseDetection::HeadPoseDetection(const std::string &pathToModel, const std::string &deviceForInference, int maxBatch, bool isBatchDynamic, bool isAsync, bool doRawOutputMessages)
 *
 * \brief Constructor
 *
 * \author  Delmiro Paes
 *
 * \param pathToModel         The path to model.
 * \param deviceForInference  The device for inference.
 * \param maxBatch            The maximum batch.
 * \param isBatchDynamic      True if is batch dynamic, false if not.
 * \param isAsync             True if is asynchronous, false if not.
 * \param doRawOutputMessages True to do raw output messages.
 **************************************************************************************************/
HeadPoseDetection::HeadPoseDetection(const std::string& pathToModel,
  const std::string& deviceForInference,
  int maxBatch, bool isBatchDynamic, bool isAsync, bool doRawOutputMessages)
  : BaseDetection("Head Pose", pathToModel, deviceForInference, maxBatch, isBatchDynamic, isAsync, doRawOutputMessages),
  outputAngleR("angle_r_fc"), outputAngleP("angle_p_fc"), outputAngleY("angle_y_fc"), enquedFaces(0) 
{
}


/**********************************************************************************************//**
 * \fn  void HeadPoseDetection::submitRequest()
 *
 * \brief Submit request
 *
 * \author  Delmiro Paes
 **************************************************************************************************/
void HeadPoseDetection::submitRequest() 
{
  if (!enquedFaces) return;
  if (isBatchDynamic) 
  {
    request->SetBatch(enquedFaces);
  }
  BaseDetection::submitRequest();
  enquedFaces = 0;
}


/**********************************************************************************************//**
 * \fn  void HeadPoseDetection::enqueue(const cv::Mat &face)
 *
 * \brief Adds an object onto the end of this queue
 *
 * \author  Delmiro Paes
 *
 * \param face  The face.
 **************************************************************************************************/
void HeadPoseDetection::enqueue(const cv::Mat& face) 
{
  if (!enabled()) 
  {
    return;
  }
  if (enquedFaces == maxBatch) 
  {
    slog::warn << "Number of detected faces more than maximum(" << maxBatch << ") processed by Head Pose estimator" << slog::endl;
    return;
  }
  if (!request) {
    request = net.CreateInferRequestPtr();
  }

  Blob::Ptr inputBlob = request->GetBlob(input);

  matU8ToBlob<uint8_t>(face, inputBlob, enquedFaces);

  enquedFaces++;
}


/**********************************************************************************************//**
 * \fn  HeadPoseDetection::Results HeadPoseDetection::operator[] (int idx) const
 *
 * \brief Array indexer operator
 *
 * \author  Delmiro Paes
 *
 * \param idx Zero-based index of the.
 *
 * \returns The indexed value.
 **************************************************************************************************/
HeadPoseDetection::Results HeadPoseDetection::operator[] (int idx) const 
{
  Blob::Ptr  angleR = request->GetBlob(outputAngleR);
  Blob::Ptr  angleP = request->GetBlob(outputAngleP);
  Blob::Ptr  angleY = request->GetBlob(outputAngleY);

  HeadPoseDetection::Results r = { angleR->buffer().as<float*>()[idx],
                                  angleP->buffer().as<float*>()[idx],
                                  angleY->buffer().as<float*>()[idx] };

  if (doRawOutputMessages) 
  {
    std::cout << "[" << idx << "] element, yaw = " << r.angle_y <<
      ", pitch = " << r.angle_p <<
      ", roll = " << r.angle_r << std::endl;
  }

  return r;
}


/**********************************************************************************************//**
 * \fn  CNNNetwork HeadPoseDetection::read()
 *
 * \brief Gets the read
 *
 * \author  Delmiro Paes
 *
 * \exception std::logic_error  Raised when a logic error condition occurs.
 *
 * \returns A CNNNetwork.
 **************************************************************************************************/
CNNNetwork HeadPoseDetection::read() 
{
  slog::info << "Loading network files for Head Pose Estimation network" << slog::endl;
  CNNNetReader netReader;
  // Read network model
  netReader.ReadNetwork(pathToModel);
  // Set maximum batch size
  netReader.getNetwork().setBatchSize(maxBatch);
  slog::info << "Batch size is set to  " << netReader.getNetwork().getBatchSize() << " for Head Pose Estimation network" << slog::endl;
  // Extract model name and load its weights
  std::string binFileName = fileNameNoExt(pathToModel) + ".bin";
  netReader.ReadWeights(binFileName);

  // ---------------------------Check inputs -------------------------------------------------------------
  slog::info << "Checking Head Pose Estimation network inputs" << slog::endl;
  InputsDataMap inputInfo(netReader.getNetwork().getInputsInfo());
  if (inputInfo.size() != 1) 
  {
    throw std::logic_error("Head Pose Estimation network should have only one input");
  }
  InputInfo::Ptr& inputInfoFirst = inputInfo.begin()->second;
  inputInfoFirst->setPrecision(Precision::U8);
  input = inputInfo.begin()->first;
  // -----------------------------------------------------------------------------------------------------

  // ---------------------------Check outputs ------------------------------------------------------------
  slog::info << "Checking Head Pose Estimation network outputs" << slog::endl;
  OutputsDataMap outputInfo(netReader.getNetwork().getOutputsInfo());
  if (outputInfo.size() != 3) 
  {
    throw std::logic_error("Head Pose Estimation network should have 3 outputs");
  }

  for (auto& output : outputInfo) 
  {
    output.second->setPrecision(Precision::FP32);
  }

  std::map<std::string, bool> layerNames = {
      {outputAngleR, false},
      {outputAngleP, false},
      {outputAngleY, false}
  };

  for (auto&& output : outputInfo) 
  {
    CNNLayerPtr layer = output.second->getCreatorLayer().lock();
    if (!layer) 
    {
      throw std::logic_error("Layer pointer is invalid");
    }

    if (layerNames.find(layer->name) == layerNames.end()) 
    {
      throw std::logic_error("Head Pose Estimation network output layer unknown: " + layer->name + ", should be " +
        outputAngleR + " or " + outputAngleP + " or " + outputAngleY);
    }

    if (layer->type != "FullyConnected") 
    {
      throw std::logic_error("Head Pose Estimation network output layer (" + layer->name + ") has invalid type: " +
        layer->type + ", should be FullyConnected");
    }

    auto fc = dynamic_cast<FullyConnectedLayer*>(layer.get());
    if (!fc) 
    {
      throw std::logic_error("Fully connected layer is not valid");
    }
    
    if (fc->_out_num != 1) 
    {
      throw std::logic_error("Head Pose Estimation network output layer (" + layer->name + ") has invalid out-size=" +
        std::to_string(fc->_out_num) + ", should be 1");
    }
    layerNames[layer->name] = true;
  }

  slog::info << "Loading Head Pose Estimation model to the " << deviceForInference << " plugin" << slog::endl;

  _enabled = true;
  return netReader.getNetwork();
}


/**********************************************************************************************//**
 * \fn  EmotionsDetection::EmotionsDetection(const std::string &pathToModel, const std::string &deviceForInference, int maxBatch, bool isBatchDynamic, bool isAsync, bool doRawOutputMessages)
 *
 * \brief Constructor
 *
 * \author  Delmiro Paes
 *
 * \param pathToModel         The path to model.
 * \param deviceForInference  The device for inference.
 * \param maxBatch            The maximum batch.
 * \param isBatchDynamic      True if is batch dynamic, false if not.
 * \param isAsync             True if is asynchronous, false if not.
 * \param doRawOutputMessages True to do raw output messages.
 **************************************************************************************************/
EmotionsDetection::EmotionsDetection(const std::string& pathToModel,
  const std::string& deviceForInference,
  int maxBatch, bool isBatchDynamic, bool isAsync, bool doRawOutputMessages)
  : BaseDetection("Emotions Recognition", pathToModel, deviceForInference, maxBatch, isBatchDynamic, isAsync, doRawOutputMessages),
  enquedFaces(0) {
}


/**********************************************************************************************//**
 * \fn  void EmotionsDetection::submitRequest()
 *
 * \brief Submit request
 *
 * \author  Delmiro Paes
 **************************************************************************************************/
void EmotionsDetection::submitRequest() 
{
  if (!enquedFaces) return;

  if (isBatchDynamic) 
  {
    request->SetBatch(enquedFaces);
  }

  BaseDetection::submitRequest();

  enquedFaces = 0;
}


/**********************************************************************************************//**
 * \fn  void EmotionsDetection::enqueue(const cv::Mat &face)
 *
 * \brief Adds an object onto the end of this queue
 *
 * \author  Delmiro Paes
 *
 * \param face  The face.
 **************************************************************************************************/
void EmotionsDetection::enqueue(const cv::Mat& face) 
{
  if (!enabled()) 
  {
    return;
  }

  if (enquedFaces == maxBatch) 
  {
    slog::warn << "Number of detected faces more than maximum(" << maxBatch << ") processed by Emotions Recognition network" << slog::endl;
    return;
  }

  if (!request) 
  {
    request = net.CreateInferRequestPtr();
  }

  Blob::Ptr inputBlob = request->GetBlob(input);

  matU8ToBlob<uint8_t>(face, inputBlob, enquedFaces);

  enquedFaces++;
}


/**********************************************************************************************//**
 * \fn  std::map<std::string, float> EmotionsDetection::operator[] (int idx) const
 *
 * \brief Array indexer operator
 *
 * \author  Delmiro Paes
 *
 * \exception std::logic_error  Raised when a logic error condition occurs.
 *
 * \param idx Zero-based index of the.
 *
 * \returns The indexed value.
 **************************************************************************************************/
std::map<std::string, float> EmotionsDetection::operator[] (int idx) const 
{
  // Vector of supported emotions
  static const std::vector<std::string> emotionsVec = { "neutral", "happy", "sad", "surprise", "anger" };
  auto emotionsVecSize = emotionsVec.size();

  Blob::Ptr emotionsBlob = request->GetBlob(outputEmotions);

  /* emotions vector must have the same size as number of channels
   * in model output. Default output format is NCHW, so index 1 is checked */
  size_t numOfChannels = emotionsBlob->getTensorDesc().getDims().at(1);

  if (numOfChannels != emotionsVecSize) 
  {
    throw std::logic_error("Output size (" + std::to_string(numOfChannels) +
      ") of the Emotions Recognition network is not equal "
      "to used emotions vector size (" +
      std::to_string(emotionsVec.size()) + ")");
  }

  auto emotionsValues = emotionsBlob->buffer().as<float*>();
  auto outputIdxPos = emotionsValues + idx * emotionsVecSize;
  std::map<std::string, float> emotions;

  if (doRawOutputMessages) 
  {
    std::cout << "[" << idx << "] element, predicted emotions (name = prob):" << std::endl;
  }

  for (size_t i = 0; i < emotionsVecSize; i++) 
  {
    emotions[emotionsVec[i]] = outputIdxPos[i];

    if (doRawOutputMessages) 
    {
      std::cout << emotionsVec[i] << " = " << outputIdxPos[i];
      if (emotionsVecSize - 1 != i) 
      {
        std::cout << ", ";
      }
      else 
      {
        std::cout << std::endl;
      }
    }
  }

  return emotions;
}


/**********************************************************************************************//**
 * \fn  CNNNetwork EmotionsDetection::read()
 *
 * \brief Gets the read
 *
 * \author  Delmiro Paes
 *
 * \exception std::logic_error  Raised when a logic error condition occurs.
 *
 * \returns A CNNNetwork.
 **************************************************************************************************/
CNNNetwork EmotionsDetection::read() 
{
  slog::info << "Loading network files for Emotions Recognition" << slog::endl;
  InferenceEngine::CNNNetReader netReader;
  // Read network model
  netReader.ReadNetwork(pathToModel);

  // Set maximum batch size
  netReader.getNetwork().setBatchSize(maxBatch);
  slog::info << "Batch size is set to " << netReader.getNetwork().getBatchSize() << " for Emotions Recognition" << slog::endl;
  
  // Extract model name and load its weights
  std::string binFileName = fileNameNoExt(pathToModel) + ".bin";
  netReader.ReadWeights(binFileName);

  // -----------------------------------------------------------------------------------------------------

  // Emotions Recognition network should have one input and one output.
  // ---------------------------Check inputs -------------------------------------------------------------
  slog::info << "Checking Emotions Recognition network inputs" << slog::endl;
  InferenceEngine::InputsDataMap inputInfo(netReader.getNetwork().getInputsInfo());
  
  if (inputInfo.size() != 1) 
  {
    throw std::logic_error("Emotions Recognition network should have only one input");
  }

  auto& inputInfoFirst = inputInfo.begin()->second;
  inputInfoFirst->setPrecision(Precision::U8);
  input = inputInfo.begin()->first;
  // -----------------------------------------------------------------------------------------------------

  // ---------------------------Check outputs ------------------------------------------------------------
  slog::info << "Checking Emotions Recognition network outputs" << slog::endl;
  InferenceEngine::OutputsDataMap outputInfo(netReader.getNetwork().getOutputsInfo());

  if (outputInfo.size() != 1) 
  {
    throw std::logic_error("Emotions Recognition network should have one output layer");
  }

  for (auto& output : outputInfo) 
  {
    output.second->setPrecision(Precision::FP32);
  }

  DataPtr emotionsOutput = outputInfo.begin()->second;

  if (!emotionsOutput) 
  {
    throw std::logic_error("Emotions output data pointer is invalid");
  }

  auto emotionsCreatorLayer = emotionsOutput->getCreatorLayer().lock();

  if (!emotionsCreatorLayer) 
  {
    throw std::logic_error("Emotions creator layer pointer is invalid");
  }

  if (emotionsCreatorLayer->type != "SoftMax") 
  {
    throw std::logic_error("In Emotions Recognition network, Emotion layer ("
      + emotionsCreatorLayer->name +
      ") should be a SoftMax, but was: " +
      emotionsCreatorLayer->type);
  }
  slog::info << "Emotions layer: " << emotionsCreatorLayer->name << slog::endl;

  outputEmotions = emotionsOutput->getName();

  slog::info << "Loading Emotions Recognition model to the " << deviceForInference << " plugin" << slog::endl;
  
  _enabled = true;

  return netReader.getNetwork();
}


/**********************************************************************************************//**
 * \fn  FacialLandmarksDetection::FacialLandmarksDetection(const std::string &pathToModel, const std::string &deviceForInference, int maxBatch, bool isBatchDynamic, bool isAsync, bool doRawOutputMessages)
 *
 * \brief Constructor
 *
 * \author  Delmiro Paes
 *
 * \param pathToModel         The path to model.
 * \param deviceForInference  The device for inference.
 * \param maxBatch            The maximum batch.
 * \param isBatchDynamic      True if is batch dynamic, false if not.
 * \param isAsync             True if is asynchronous, false if not.
 * \param doRawOutputMessages True to do raw output messages.
 **************************************************************************************************/
FacialLandmarksDetection::FacialLandmarksDetection(const std::string& pathToModel,
  const std::string& deviceForInference,
  int maxBatch, bool isBatchDynamic, bool isAsync, bool doRawOutputMessages)
  : BaseDetection("Facial Landmarks", pathToModel, deviceForInference, maxBatch, isBatchDynamic, isAsync, doRawOutputMessages),
  outputFacialLandmarksBlobName("align_fc3"), enquedFaces(0) 
{
}


/**********************************************************************************************//**
 * \fn  void FacialLandmarksDetection::submitRequest()
 *
 * \brief Submit request
 *
 * \author  Delmiro Paes
 **************************************************************************************************/
void FacialLandmarksDetection::submitRequest() 
{
  if (!enquedFaces) return;
  if (isBatchDynamic) 
  {
    request->SetBatch(enquedFaces);
  }
  BaseDetection::submitRequest();
  enquedFaces = 0;
}


/**********************************************************************************************//**
 * \fn  void FacialLandmarksDetection::enqueue(const cv::Mat &face)
 *
 * \brief Adds an object onto the end of this queue
 *
 * \author  Delmiro Paes
 *
 * \param face  The face.
 **************************************************************************************************/
void FacialLandmarksDetection::enqueue(const cv::Mat& face) 
{
  if (!enabled()) 
  {
    return;
  }

  if (enquedFaces == maxBatch) 
  {
    slog::warn << "Number of detected faces more than maximum(" << maxBatch << ") processed by Facial Landmarks estimator" << slog::endl;
    return;
  }

  if (!request) 
  {
    request = net.CreateInferRequestPtr();
  }

  Blob::Ptr inputBlob = request->GetBlob(input);

  matU8ToBlob<uint8_t>(face, inputBlob, enquedFaces);

  enquedFaces++;
}


/**********************************************************************************************//**
 * \fn  std::vector<float> FacialLandmarksDetection::operator[] (int idx) const
 *
 * \brief Array indexer operator
 *
 * \author  Delmiro Paes
 *
 * \param idx Zero-based index of the.
 *
 * \returns The indexed value.
 **************************************************************************************************/
std::vector<float> FacialLandmarksDetection::operator[] (int idx) const 
{
  std::vector<float> normedLandmarks;

  auto landmarksBlob = request->GetBlob(outputFacialLandmarksBlobName);
  auto n_lm = getTensorChannels(landmarksBlob->getTensorDesc());
  const float* normed_coordinates = request->GetBlob(outputFacialLandmarksBlobName)->buffer().as<float*>();

  if (doRawOutputMessages) 
  {
    std::cout << "[" << idx << "] element, normed facial landmarks coordinates (x, y):" << std::endl;
  }

  auto begin = n_lm * idx;
  auto end = begin + n_lm / 2;
  for (auto i_lm = begin; i_lm < end; ++i_lm) 
  {
    float normed_x = normed_coordinates[2 * i_lm];
    float normed_y = normed_coordinates[2 * i_lm + 1];

    if (doRawOutputMessages) 
    {
      std::cout << normed_x << ", " << normed_y << std::endl;
    }

    normedLandmarks.push_back(normed_x);
    normedLandmarks.push_back(normed_y);
  }

  return normedLandmarks;
}


/**********************************************************************************************//**
 * \fn  CNNNetwork FacialLandmarksDetection::read()
 *
 * \brief Gets the read
 *
 * \author  Delmiro Paes
 *
 * \exception std::logic_error  Raised when a logic error condition occurs.
 *
 * \returns A CNNNetwork.
 **************************************************************************************************/
CNNNetwork FacialLandmarksDetection::read() 
{
  slog::info << "Loading network files for Facial Landmarks Estimation" << slog::endl;
  CNNNetReader netReader;
  // Read network model
  netReader.ReadNetwork(pathToModel);
  // Set maximum batch size
  netReader.getNetwork().setBatchSize(maxBatch);
  slog::info << "Batch size is set to  " << netReader.getNetwork().getBatchSize() << " for Facial Landmarks Estimation network" << slog::endl;
  // Extract model name and load its weights
  std::string binFileName = fileNameNoExt(pathToModel) + ".bin";
  netReader.ReadWeights(binFileName);

  // ---------------------------Check inputs -------------------------------------------------------------
  slog::info << "Checking Facial Landmarks Estimation network inputs" << slog::endl;
  InputsDataMap inputInfo(netReader.getNetwork().getInputsInfo());
  if (inputInfo.size() != 1) 
  {
    throw std::logic_error("Facial Landmarks Estimation network should have only one input");
  }
  InputInfo::Ptr& inputInfoFirst = inputInfo.begin()->second;
  inputInfoFirst->setPrecision(Precision::U8);
  input = inputInfo.begin()->first;
  // -----------------------------------------------------------------------------------------------------

  // ---------------------------Check outputs ------------------------------------------------------------
  slog::info << "Checking Facial Landmarks Estimation network outputs" << slog::endl;
  OutputsDataMap outputInfo(netReader.getNetwork().getOutputsInfo());
  if (outputInfo.size() != 1) 
  {
    throw std::logic_error("Facial Landmarks Estimation network should have only one output");
  }

  for (auto& output : outputInfo) 
  {
    output.second->setPrecision(Precision::FP32);
  }

  std::map<std::string, bool> layerNames = { {outputFacialLandmarksBlobName, false} };

  for (auto&& output : outputInfo) 
  {
    CNNLayerPtr layer = output.second->getCreatorLayer().lock();
    if (!layer) 
    {
      throw std::logic_error("Layer pointer is invalid");
    }

    if (layerNames.find(layer->name) == layerNames.end()) 
    {
      throw std::logic_error("Facial Landmarks Estimation network output layer unknown: " + layer->name + ", should be " +
        outputFacialLandmarksBlobName);
    }

    const SizeVector outputDims = output.second->getTensorDesc().getDims();
    if (outputDims[1] != 70) 
    {
      throw std::logic_error("Facial Landmarks Estimation network output layer should have 70 as a last dimension");
    }
    layerNames[layer->name] = true;
  }

  slog::info << "Loading Facial Landmarks Estimation model to the " << deviceForInference << " plugin" << slog::endl;

  _enabled = true;
  return netReader.getNetwork();
}


/**********************************************************************************************//**
 * \fn  Load::Load(BaseDetection& detector)
 *
 * \brief Constructor
 *
 * \author  Delmiro Paes
 *
 * \param [in,out]  detector  The detector to load.
 **************************************************************************************************/
Load::Load(BaseDetection& detector) : detector(detector) {
}


/**********************************************************************************************//**
 * \fn  void Load::into(InferenceEngine::Core & ie, const std::string & deviceName, bool enable_dynamic_batch) const
 *
 * \brief Intoes
 *
 * \author  Delmiro Paes
 *
 * \param [in,out]  ie                    The IE.
 * \param           deviceName            Name of the device.
 * \param           enable_dynamic_batch  True to enable, false to disable the dynamic batch.
 **************************************************************************************************/
void Load::into(InferenceEngine::Core& ie, const std::string& deviceName, bool enable_dynamic_batch) const 
{
  if (detector.enabled()) 
  {
    std::map<std::string, std::string> config = { };
    bool isPossibleDynBatch = deviceName.find("CPU") != std::string::npos || deviceName.find("GPU") != std::string::npos;

    if (enable_dynamic_batch && isPossibleDynBatch) 
    {
      config[PluginConfigParams::KEY_DYN_BATCH_ENABLED] = PluginConfigParams::YES;
    }

    detector.net = ie.LoadNetwork(detector.read(), deviceName, config);
  }
}


/**********************************************************************************************//**
 * \fn  CallStat::CallStat()
 *
 * \brief Default constructor
 *
 * \author  Delmiro Paes
 **************************************************************************************************/
CallStat::CallStat() : _number_of_calls(0), _total_duration(0.0), _last_call_duration(0.0), _smoothed_duration(-1.0) 
{
}


/**********************************************************************************************//**
 * \fn  double CallStat::getSmoothedDuration()
 *
 * \brief Gets smoothed duration
 *
 * \author  Delmiro Paes
 *
 * \returns The smoothed duration.
 **************************************************************************************************/
double CallStat::getSmoothedDuration() 
{
  // Additional check is needed for the first frame while duration of the first
  // visualisation is not calculated yet.
  if (_smoothed_duration < 0) 
  {
    auto t = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<ms>(t - _last_call_start).count();
  }

  return _smoothed_duration;
}


/**********************************************************************************************//**
 * \fn  double CallStat::getTotalDuration()
 *
 * \brief Gets total duration
 *
 * \author  Delmiro Paes
 *
 * \returns The total duration.
 **************************************************************************************************/
double CallStat::getTotalDuration() 
{
  return _total_duration;
}


/**********************************************************************************************//**
 * \fn  double CallStat::getLastCallDuration()
 *
 * \brief Gets the last call duration
 *
 * \author  Delmiro Paes
 *
 * \returns The last call duration.
 **************************************************************************************************/
double CallStat::getLastCallDuration() 
{
  return _last_call_duration;
}


/**********************************************************************************************//**
 * \fn  void CallStat::calculateDuration()
 *
 * \brief Calculates the duration
 *
 * \author  Delmiro Paes
 **************************************************************************************************/
void CallStat::calculateDuration() 
{
  auto t = std::chrono::high_resolution_clock::now();
  _last_call_duration = std::chrono::duration_cast<ms>(t - _last_call_start).count();
  _number_of_calls++;
  _total_duration += _last_call_duration;

  if (_smoothed_duration < 0) 
  {
    _smoothed_duration = _last_call_duration;
  }
  
  double alpha = 0.1;
  _smoothed_duration = _smoothed_duration * (1.0 - alpha) + _last_call_duration * alpha;
}


/**********************************************************************************************//**
 * \fn  void CallStat::setStartTime()
 *
 * \brief Sets start time
 *
 * \author  Delmiro Paes
 **************************************************************************************************/
void CallStat::setStartTime() 
{
  _last_call_start = std::chrono::high_resolution_clock::now();
}


/**********************************************************************************************//**
 * \fn  void Timer::start(const std::string& name)
 *
 * \brief Starts the given name
 *
 * \author  Delmiro Paes
 *
 * \param name  The name.
 **************************************************************************************************/
void Timer::start(const std::string& name) 
{
  if (_timers.find(name) == _timers.end()) 
  {
    _timers[name] = CallStat();
  }

  _timers[name].setStartTime();
}


/**********************************************************************************************//**
 * \fn  void Timer::finish(const std::string& name)
 *
 * \brief Finishes the given name
 *
 * \author  Delmiro Paes
 *
 * \param name  The name.
 **************************************************************************************************/
void Timer::finish(const std::string& name) 
{
  auto& timer = (*this)[name];
  timer.calculateDuration();
}


/**********************************************************************************************//**
 * \fn  CallStat& Timer::operator[](const std::string& name)
 *
 * \brief Array indexer operator
 *
 * \author  Delmiro Paes
 *
 * \exception std::logic_error  Raised when a logic error condition occurs.
 *
 * \param name  The name.
 *
 * \returns The indexed value.
 **************************************************************************************************/
CallStat& Timer::operator[](const std::string& name) 
{
  if (_timers.find(name) == _timers.end()) 
  {
    throw std::logic_error("No timer with name " + name + ".");
  }

  return _timers[name];
}
