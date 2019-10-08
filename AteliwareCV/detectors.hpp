//
// Copyright (C) 2019 Ateliware
//

/************************************************************************************************
 * \file  detectors.hpp.
 *
 * \brief Implements the detectors class
 **************************************************************************************************/

# pragma once

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

#include <samples/common.hpp>
#include <samples/slog.hpp>

#include <ie_iextension.h>
#ifdef WITH_EXTENSIONS
#include <ext_list.hpp>
#endif

#include <opencv2/opencv.hpp>

 // -------------------------Generic routines for detection networks-------------------------------------------------


 /**********************************************************************************************//**
  * \struct  BaseDetection
  *
  * \brief A base detection.
  *
  * \author  Delmiro Paes
  **************************************************************************************************/
struct BaseDetection {
  InferenceEngine::ExecutableNetwork net;
  /** \brief The request */
  InferenceEngine::InferRequest::Ptr request;
  /** \brief Name of the topo */
  std::string topoName;
  /** \brief The path to model */
  std::string pathToModel;
  /** \brief The device for inference */
  std::string deviceForInference;
  /** \brief The maximum batch */
  const size_t maxBatch;
  /** \brief True if is batch dynamic, false if not */
  bool isBatchDynamic;
  /** \brief True if is asynchronous, false if not */
  const bool isAsync;


  /**********************************************************************************************//**
   * \property  mutable bool enablingChecked
   *
   * \brief Gets a value indicating whether the enabling checked
   *
   * \returns True if enabling checked, false if not.
   **************************************************************************************************/
  mutable bool enablingChecked;


  /**********************************************************************************************//**
   * \property  mutable bool _enabled
   *
   * \brief Gets a value indicating whether this object is enabled
   *
   * \returns True if enabled, false if not.
   **************************************************************************************************/
  mutable bool _enabled;
  /** \brief True to do raw output messages */
  const bool doRawOutputMessages;


  /**********************************************************************************************//**
   * \fn  BaseDetection(std::string topoName, const std::string &pathToModel, const std::string &deviceForInference, int maxBatch, bool isBatchDynamic, bool isAsync, bool doRawOutputMessages);
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
  BaseDetection(std::string topoName,
    const std::string& pathToModel,
    const std::string& deviceForInference,
    int maxBatch, bool isBatchDynamic, bool isAsync,
    bool doRawOutputMessages);


  /**********************************************************************************************//**
   * \fn  virtual ~BaseDetection();
   *
   * \brief Destructor
   *
   * \author  Delmiro Paes
   **************************************************************************************************/
  virtual ~BaseDetection();


  /**********************************************************************************************//**
   * \fn  InferenceEngine::ExecutableNetwork* operator->();
   *
   * \brief Member dereference operator
   *
   * \author  Delmiro Paes
   *
   * \returns The dereferenced object.
   **************************************************************************************************/
  InferenceEngine::ExecutableNetwork* operator ->();


  /**********************************************************************************************//**
   * \fn  virtual InferenceEngine::CNNNetwork read() = 0;
   *
   * \brief Gets the read
   *
   * \author  Delmiro Paes
   *
   * \returns An InferenceEngine::CNNNetwork.
   **************************************************************************************************/
  virtual InferenceEngine::CNNNetwork read() = 0;


  /**********************************************************************************************//**
   * \fn  virtual void submitRequest();
   *
   * \brief Submit request
   *
   * \author  Delmiro Paes
   **************************************************************************************************/
  virtual void submitRequest();


  /**********************************************************************************************//**
   * \fn  virtual void wait();
   *
   * \brief Waits this object
   *
   * \author  Delmiro Paes
   **************************************************************************************************/
  virtual void wait();


  /**********************************************************************************************//**
   * \fn  bool enabled() const;
   *
   * \brief Enabled this object
   *
   * \author  Delmiro Paes
   *
   * \returns True if it succeeds, false if it fails.
   **************************************************************************************************/
  bool enabled() const;


  /**********************************************************************************************//**
   * \fn  void printPerformanceCounts(std::string fullDeviceName);
   *
   * \brief Print performance counts
   *
   * \author  Delmiro Paes
   *
   * \param fullDeviceName  Name of the full device.
   **************************************************************************************************/
  void printPerformanceCounts(std::string fullDeviceName);
};


/**********************************************************************************************//**
 * \struct  FaceDetection
 *
 * \brief A face detection.
 *
 * \author  Delmiro Paes
 **************************************************************************************************/
struct FaceDetection : BaseDetection {
  struct Result {
    int label;
    float confidence;
    cv::Rect location;
  };

  /** \brief The input */
  std::string input;
  /** \brief The output */
  std::string output;
  /** \brief The detection threshold */
  double detectionThreshold;
  /** \brief Number of maximum proposals */
  int maxProposalCount;
  /** \brief Size of the object */
  int objectSize;
  /** \brief The enqued frames */
  int enquedFrames;
  /** \brief The width */
  float width;
  /** \brief The height */
  float height;
  /** \brief The bb enlarge coefficient */
  float bb_enlarge_coefficient;
  /** \brief The bb dx coefficient */
  float bb_dx_coefficient;
  /** \brief The bb dy coefficient */
  float bb_dy_coefficient;
  /** \brief True if results fetched */
  bool resultsFetched;
  /** \brief The labels */
  std::vector<std::string> labels;
  /** \brief The results */
  std::vector<Result> results;


  /**********************************************************************************************//**
   * \fn  FaceDetection(const std::string &pathToModel, const std::string &deviceForInference, int maxBatch, bool isBatchDynamic, bool isAsync, double detectionThreshold, bool doRawOutputMessages, float bb_enlarge_coefficient, float bb_dx_coefficient, float bb_dy_coefficient);
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
  FaceDetection(const std::string& pathToModel,
    const std::string& deviceForInference,
    int maxBatch, bool isBatchDynamic, bool isAsync,
    double detectionThreshold, bool doRawOutputMessages,
    float bb_enlarge_coefficient, float bb_dx_coefficient,
    float bb_dy_coefficient);


  /**********************************************************************************************//**
   * \fn  InferenceEngine::CNNNetwork read() override;
   *
   * \brief Gets the read
   *
   * \author  Delmiro Paes
   *
   * \returns An InferenceEngine::CNNNetwork.
   **************************************************************************************************/
  InferenceEngine::CNNNetwork read() override;


  /**********************************************************************************************//**
   * \fn  void submitRequest() override;
   *
   * \brief Submit request
   *
   * \author  Delmiro Paes
   **************************************************************************************************/
  void submitRequest() override;


  /**********************************************************************************************//**
   * \fn  void enqueue(const cv::Mat &frame);
   *
   * \brief Adds an object onto the end of this queue
   *
   * \author  Delmiro Paes
   *
   * \param frame The frame.
   **************************************************************************************************/
  void enqueue(const cv::Mat& frame);


  /**********************************************************************************************//**
   * \fn  void fetchResults();
   *
   * \brief Fetches the results
   *
   * \author  Delmiro Paes
   **************************************************************************************************/
  void fetchResults();
};


/**********************************************************************************************//**
 * \struct  AgeGenderDetection
 *
 * \brief An age gender detection.
 *
 * \author  Delmiro Paes
 **************************************************************************************************/
struct AgeGenderDetection : BaseDetection {
  struct Result {
    float age;
    float maleProb;
  };

  /** \brief The input */
  std::string input;
  /** \brief The output age */
  std::string outputAge;
  /** \brief The output gender */
  std::string outputGender;
  /** \brief The enqued faces */
  size_t enquedFaces;


  /**********************************************************************************************//**
   * \fn  AgeGenderDetection(const std::string &pathToModel, const std::string &deviceForInference, int maxBatch, bool isBatchDynamic, bool isAsync, bool doRawOutputMessages);
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
  AgeGenderDetection(const std::string& pathToModel,
    const std::string& deviceForInference,
    int maxBatch, bool isBatchDynamic, bool isAsync,
    bool doRawOutputMessages);


  /**********************************************************************************************//**
   * \fn  InferenceEngine::CNNNetwork read() override;
   *
   * \brief Gets the read
   *
   * \author  Delmiro Paes
   *
   * \returns An InferenceEngine::CNNNetwork.
   **************************************************************************************************/
  InferenceEngine::CNNNetwork read() override;


  /**********************************************************************************************//**
   * \fn  void submitRequest() override;
   *
   * \brief Submit request
   *
   * \author  Delmiro Paes
   **************************************************************************************************/
  void submitRequest() override;


  /**********************************************************************************************//**
   * \fn  void enqueue(const cv::Mat &face);
   *
   * \brief Adds an object onto the end of this queue
   *
   * \author  Delmiro Paes
   *
   * \param face  The face.
   **************************************************************************************************/
  void enqueue(const cv::Mat& face);


  /**********************************************************************************************//**
   * \fn  Result operator[] (int idx) const;
   *
   * \brief Array indexer operator
   *
   * \author  Delmiro Paes
   *
   * \param idx Zero-based index of the.
   *
   * \returns The indexed value.
   **************************************************************************************************/
  Result operator[] (int idx) const;
};


/**********************************************************************************************//**
 * \struct  HeadPoseDetection
 *
 * \brief A head pose detection.
 *
 * \author  Delmiro Paes
 **************************************************************************************************/
struct HeadPoseDetection : BaseDetection {
  struct Results {
    float angle_r;
    float angle_p;
    float angle_y;
  };

  /** \brief The input */
  std::string input;
  /** \brief The output angle r */
  std::string outputAngleR;
  /** \brief The output angle p */
  std::string outputAngleP;
  /** \brief The output angle y coordinate */
  std::string outputAngleY;
  /** \brief The enqued faces */
  size_t enquedFaces;
  /** \brief The camera matrix */
  cv::Mat cameraMatrix;


  /**********************************************************************************************//**
   * \fn  HeadPoseDetection(const std::string &pathToModel, const std::string &deviceForInference, int maxBatch, bool isBatchDynamic, bool isAsync, bool doRawOutputMessages);
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
  HeadPoseDetection(const std::string& pathToModel,
    const std::string& deviceForInference,
    int maxBatch, bool isBatchDynamic, bool isAsync,
    bool doRawOutputMessages);


  /**********************************************************************************************//**
   * \fn  InferenceEngine::CNNNetwork read() override;
   *
   * \brief Gets the read
   *
   * \author  Delmiro Paes
   *
   * \returns An InferenceEngine::CNNNetwork.
   **************************************************************************************************/
  InferenceEngine::CNNNetwork read() override;


  /**********************************************************************************************//**
   * \fn  void submitRequest() override;
   *
   * \brief Submit request
   *
   * \author  Delmiro Paes
   **************************************************************************************************/
  void submitRequest() override;


  /**********************************************************************************************//**
   * \fn  void enqueue(const cv::Mat &face);
   *
   * \brief Adds an object onto the end of this queue
   *
   * \author  Delmiro Paes
   *
   * \param face  The face.
   **************************************************************************************************/
  void enqueue(const cv::Mat& face);


  /**********************************************************************************************//**
   * \fn  Results operator[] (int idx) const;
   *
   * \brief Array indexer operator
   *
   * \author  Delmiro Paes
   *
   * \param idx Zero-based index of the.
   *
   * \returns The indexed value.
   **************************************************************************************************/
  Results operator[] (int idx) const;
};


/**********************************************************************************************//**
 * \struct  EmotionsDetection
 *
 * \brief The emotions detection.
 *
 * \author  Delmiro Paes
 **************************************************************************************************/
struct EmotionsDetection : BaseDetection {
  std::string input;
  /** \brief The output emotions */
  std::string outputEmotions;
  /** \brief The enqued faces */
  size_t enquedFaces;


  /**********************************************************************************************//**
   * \fn  EmotionsDetection(const std::string &pathToModel, const std::string &deviceForInference, int maxBatch, bool isBatchDynamic, bool isAsync, bool doRawOutputMessages);
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
  EmotionsDetection(const std::string& pathToModel,
    const std::string& deviceForInference,
    int maxBatch, bool isBatchDynamic, bool isAsync,
    bool doRawOutputMessages);


  /**********************************************************************************************//**
   * \fn  InferenceEngine::CNNNetwork read() override;
   *
   * \brief Gets the read
   *
   * \author  Delmiro Paes
   *
   * \returns An InferenceEngine::CNNNetwork.
   **************************************************************************************************/
  InferenceEngine::CNNNetwork read() override;


  /**********************************************************************************************//**
   * \fn  void submitRequest() override;
   *
   * \brief Submit request
   *
   * \author  Delmiro Paes
   **************************************************************************************************/
  void submitRequest() override;


  /**********************************************************************************************//**
   * \fn  void enqueue(const cv::Mat &face);
   *
   * \brief Adds an object onto the end of this queue
   *
   * \author  Delmiro Paes
   *
   * \param face  The face.
   **************************************************************************************************/
  void enqueue(const cv::Mat& face);


  /**********************************************************************************************//**
   * \fn  std::map<std::string, float> operator[] (int idx) const;
   *
   * \brief Array indexer operator
   *
   * \author  Delmiro Paes
   *
   * \param idx Zero-based index of the.
   *
   * \returns The indexed value.
   **************************************************************************************************/
  std::map<std::string, float> operator[] (int idx) const;

  /** \brief The emotions vector */
  const std::vector<std::string> emotionsVec = { "neutral", "happy", "sad", "surprise", "anger" };
};


/**********************************************************************************************//**
 * \struct  FacialLandmarksDetection
 *
 * \brief A facial landmarks detection.
 *
 * \author  Delmiro Paes
 **************************************************************************************************/
struct FacialLandmarksDetection : BaseDetection {
  std::string input;
  /** \brief Name of the output facial landmarks BLOB */
  std::string outputFacialLandmarksBlobName;
  /** \brief The enqued faces */
  size_t enquedFaces;
  /** \brief The landmarks results */
  std::vector<std::vector<float>> landmarks_results;
  /** \brief The faces bounding boxes */
  std::vector<cv::Rect> faces_bounding_boxes;


  /**********************************************************************************************//**
   * \fn  FacialLandmarksDetection(const std::string &pathToModel, const std::string &deviceForInference, int maxBatch, bool isBatchDynamic, bool isAsync, bool doRawOutputMessages);
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
  FacialLandmarksDetection(const std::string& pathToModel,
    const std::string& deviceForInference,
    int maxBatch, bool isBatchDynamic, bool isAsync,
    bool doRawOutputMessages);


  /**********************************************************************************************//**
   * \fn  InferenceEngine::CNNNetwork read() override;
   *
   * \brief Gets the read
   *
   * \author  Delmiro Paes
   *
   * \returns An InferenceEngine::CNNNetwork.
   **************************************************************************************************/
  InferenceEngine::CNNNetwork read() override;


  /**********************************************************************************************//**
   * \fn  void submitRequest() override;
   *
   * \brief Submit request
   *
   * \author  Delmiro Paes
   **************************************************************************************************/
  void submitRequest() override;


  /**********************************************************************************************//**
   * \fn  void enqueue(const cv::Mat &face);
   *
   * \brief Adds an object onto the end of this queue
   *
   * \author  Delmiro Paes
   *
   * \param face  The face.
   **************************************************************************************************/
  void enqueue(const cv::Mat& face);


  /**********************************************************************************************//**
   * \fn  std::vector<float> operator[] (int idx) const;
   *
   * \brief Array indexer operator
   *
   * \author  Delmiro Paes
   *
   * \param idx Zero-based index of the.
   *
   * \returns The indexed value.
   **************************************************************************************************/
  std::vector<float> operator[] (int idx) const;
};


/**********************************************************************************************//**
 * \struct  Load
 *
 * \brief A load.
 *
 * \author  Delmiro Paes
 **************************************************************************************************/
struct Load {
  BaseDetection& detector;


  /**********************************************************************************************//**
   * \fn  explicit Load(BaseDetection& detector);
   *
   * \brief Constructor
   *
   * \author  Delmiro Paes
   *
   * \param [in,out]  detector  The detector to load.
   **************************************************************************************************/
  explicit Load(BaseDetection& detector);


  /**********************************************************************************************//**
   * \fn  void into(InferenceEngine::Core & ie, const std::string & deviceName, bool enable_dynamic_batch = false) const;
   *
   * \brief Intoes
   *
   * \author  Delmiro Paes
   *
   * \param [in,out]  ie                    The IE.
   * \param           deviceName            Name of the device.
   * \param           enable_dynamic_batch  (Optional) True to enable, false to disable the dynamic
   *                                        batch.
   **************************************************************************************************/
  void into(InferenceEngine::Core& ie, const std::string& deviceName, bool enable_dynamic_batch = false) const;
};


/**********************************************************************************************//**
 * \class CallStat
 *
 * \brief A call stat.
 *
 * \author  Delmiro Paes
 **************************************************************************************************/
class CallStat {
public:

  /**********************************************************************************************//**
   * \typedef std::chrono::duration<double, std::ratio<1, 1000>> ms
   *
   * \brief Defines an alias representing the milliseconds
   **************************************************************************************************/
  typedef std::chrono::duration<double, std::ratio<1, 1000>> ms;


  /**********************************************************************************************//**
   * \fn  CallStat::CallStat();
   *
   * \brief Default constructor
   *
   * \author  Delmiro Paes
   **************************************************************************************************/
  CallStat();


  /**********************************************************************************************//**
   * \fn  double CallStat::getSmoothedDuration();
   *
   * \brief Gets smoothed duration
   *
   * \author  Delmiro Paes
   *
   * \returns The smoothed duration.
   **************************************************************************************************/
  double getSmoothedDuration();


  /**********************************************************************************************//**
   * \fn  double CallStat::getTotalDuration();
   *
   * \brief Gets total duration
   *
   * \author  Delmiro Paes
   *
   * \returns The total duration.
   **************************************************************************************************/
  double getTotalDuration();


  /**********************************************************************************************//**
   * \fn  double CallStat::getLastCallDuration();
   *
   * \brief Gets the last call duration
   *
   * \author  Delmiro Paes
   *
   * \returns The last call duration.
   **************************************************************************************************/
  double getLastCallDuration();


  /**********************************************************************************************//**
   * \fn  void CallStat::calculateDuration();
   *
   * \brief Calculates the duration
   *
   * \author  Delmiro Paes
   **************************************************************************************************/
  void calculateDuration();


  /**********************************************************************************************//**
   * \fn  void CallStat::setStartTime();
   *
   * \brief Sets start time
   *
   * \author  Delmiro Paes
   **************************************************************************************************/
  void setStartTime();

private:
  /** \brief Number of calls */
  size_t _number_of_calls;
  /** \brief Duration of the total */
  double _total_duration;
  /** \brief Duration of the last call */
  double _last_call_duration;
  /** \brief Duration of the smoothed */
  double _smoothed_duration;
  /** \brief The last call start */
  std::chrono::time_point<std::chrono::high_resolution_clock> _last_call_start;
};


/**********************************************************************************************//**
 * \class Timer
 *
 * \brief A timer.
 *
 * \author  Delmiro Paes
 **************************************************************************************************/
class Timer {
public:

  /**********************************************************************************************//**
   * \fn  void Timer::start(const std::string& name);
   *
   * \brief Starts the given name
   *
   * \author  Delmiro Paes
   *
   * \param name  The name.
   **************************************************************************************************/
  void start(const std::string& name);


  /**********************************************************************************************//**
   * \fn  void Timer::finish(const std::string& name);
   *
   * \brief Finishes the given name
   *
   * \author  Delmiro Paes
   *
   * \param name  The name.
   **************************************************************************************************/
  void finish(const std::string& name);


  /**********************************************************************************************//**
   * \fn  CallStat& Timer::operator[](const std::string& name);
   *
   * \brief Array indexer operator
   *
   * \author  Delmiro Paes
   *
   * \param name  The name.
   *
   * \returns The indexed value.
   **************************************************************************************************/
  CallStat& operator[](const std::string& name);

private:
  /** \brief The timers */
  std::map<std::string, CallStat> _timers;
};
