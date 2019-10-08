//
// Copyright (C) 2019 Ateliware
//

/************************************************************************************************
 * \file  face.hpp.
 *
 * \brief Implements the face class
 **************************************************************************************************/

# pragma once
#include <string>
#include <map>
#include <memory>
#include <utility>
#include <list>
#include <vector>
#include <opencv2/opencv.hpp>

#include "detectors.hpp"

 // -------------------------Describe detected face on a frame-------------------------------------------------


 /**********************************************************************************************//**
  * \struct  Face
  *
  * \brief A face.
  *
  * \author  Delmiro Paes
  **************************************************************************************************/
struct Face 
{
public:
  /** \brief The pointer */
  using Ptr = std::shared_ptr<Face>;


  /**********************************************************************************************//**
   * \fn  explicit Face(size_t id, cv::Rect& location);
   *
   * \brief Constructor
   *
   * \author  Delmiro Paes
   *
   * \param           id        The identifier.
   * \param [in,out]  location  The location.
   **************************************************************************************************/
  explicit Face(size_t id, cv::Rect& location);


  /**********************************************************************************************//**
   * \fn  void updateAge(float value);
   *
   * \brief Updates the age described by value
   *
   * \author  Delmiro Paes
   *
   * \param value The value.
   **************************************************************************************************/
  void updateAge(float value);


  /**********************************************************************************************//**
   * \fn  void updateGender(float value);
   *
   * \brief Updates the gender described by value
   *
   * \author  Delmiro Paes
   *
   * \param value The value.
   **************************************************************************************************/
  void updateGender(float value);


  /**********************************************************************************************//**
   * \fn  void updateEmotions(std::map<std::string, float> values);
   *
   * \brief Updates the emotions described by values
   *
   * \author  Delmiro Paes
   *
   * \param values  The values.
   **************************************************************************************************/
  void updateEmotions(std::map<std::string, float> values);


  /**********************************************************************************************//**
   * \fn  void updateHeadPose(HeadPoseDetection::Results values);
   *
   * \brief Updates the head pose described by values
   *
   * \author  Delmiro Paes
   *
   * \param values  The values.
   **************************************************************************************************/
  void updateHeadPose(HeadPoseDetection::Results values);


  /**********************************************************************************************//**
   * \fn  void updateLandmarks(std::vector<float> values);
   *
   * \brief Updates the landmarks described by values
   *
   * \author  Delmiro Paes
   *
   * \param values  The values.
   **************************************************************************************************/
  void updateLandmarks(std::vector<float> values);


  /**********************************************************************************************//**
   * \fn  int getAge();
   *
   * \brief Gets the age
   *
   * \author  Delmiro Paes
   *
   * \returns The age.
   **************************************************************************************************/
  int getAge();


  /**********************************************************************************************//**
   * \fn  bool isMale();
   *
   * \brief Query if this object is male
   *
   * \author  Delmiro Paes
   *
   * \returns True if male, false if not.
   **************************************************************************************************/
  bool isMale();


  /**********************************************************************************************//**
   * \fn  std::map<std::string, float> getEmotions();
   *
   * \brief Gets the emotions
   *
   * \author  Delmiro Paes
   *
   * \returns The emotions.
   **************************************************************************************************/
  std::map<std::string, float> getEmotions();


  /**********************************************************************************************//**
   * \fn  std::pair<std::string, float> getMainEmotion();
   *
   * \brief Gets main emotion
   *
   * \author  Delmiro Paes
   *
   * \returns The main emotion.
   **************************************************************************************************/
  std::pair<std::string, float> getMainEmotion();


  /**********************************************************************************************//**
   * \fn  HeadPoseDetection::Results getHeadPose();
   *
   * \brief Gets head pose
   *
   * \author  Delmiro Paes
   *
   * \returns The head pose.
   **************************************************************************************************/
  HeadPoseDetection::Results getHeadPose();


  /**********************************************************************************************//**
   * \fn  const std::vector<float>& getLandmarks();
   *
   * \brief Gets the landmarks
   *
   * \author  Delmiro Paes
   *
   * \returns The landmarks.
   **************************************************************************************************/
  const std::vector<float>& getLandmarks();


  /**********************************************************************************************//**
   * \fn  size_t getId();
   *
   * \brief Gets the identifier
   *
   * \author  Delmiro Paes
   *
   * \returns The identifier.
   **************************************************************************************************/
  size_t getId();


  /**********************************************************************************************//**
   * \fn  void ageGenderEnable(bool value);
   *
   * \brief Age gender enable
   *
   * \author  Delmiro Paes
   *
   * \param value True to value.
   **************************************************************************************************/
  void ageGenderEnable(bool value);


  /**********************************************************************************************//**
   * \fn  void emotionsEnable(bool value);
   *
   * \brief Emotions enable
   *
   * \author  Delmiro Paes
   *
   * \param value True to value.
   **************************************************************************************************/
  void emotionsEnable(bool value);


  /**********************************************************************************************//**
   * \fn  void headPoseEnable(bool value);
   *
   * \brief Head pose enable
   *
   * \author  Delmiro Paes
   *
   * \param value True to value.
   **************************************************************************************************/
  void headPoseEnable(bool value);


  /**********************************************************************************************//**
   * \fn  void landmarksEnable(bool value);
   *
   * \brief Landmarks enable
   *
   * \author  Delmiro Paes
   *
   * \param value True to value.
   **************************************************************************************************/
  void landmarksEnable(bool value);


  /**********************************************************************************************//**
   * \fn  bool isAgeGenderEnabled();
   *
   * \brief Queries if the age gender is enabled
   *
   * \author  Delmiro Paes
   *
   * \returns True if the age gender is enabled, false if not.
   **************************************************************************************************/
  bool isAgeGenderEnabled();


  /**********************************************************************************************//**
   * \fn  bool isEmotionsEnabled();
   *
   * \brief Queries if the emotions is enabled
   *
   * \author  Delmiro Paes
   *
   * \returns True if the emotions is enabled, false if not.
   **************************************************************************************************/
  bool isEmotionsEnabled();


  /**********************************************************************************************//**
   * \fn  bool isHeadPoseEnabled();
   *
   * \brief Queries if the head pose is enabled
   *
   * \author  Delmiro Paes
   *
   * \returns True if the head pose is enabled, false if not.
   **************************************************************************************************/
  bool isHeadPoseEnabled();


  /**********************************************************************************************//**
   * \fn  bool isLandmarksEnabled();
   *
   * \brief Queries if the landmarks is enabled
   *
   * \author  Delmiro Paes
   *
   * \returns True if the landmarks is enabled, false if not.
   **************************************************************************************************/
  bool isLandmarksEnabled();

public:
  /** \brief The location */
  cv::Rect _location;
  /** \brief The intensity mean */
  float _intensity_mean;

private:
  /** \brief The identifier */
  size_t _id;
  /** \brief The age */
  float _age;
  /** \brief The male score */
  float _maleScore;
  /** \brief The female score */
  float _femaleScore;
  /** \brief The emotions */
  std::map<std::string, float> _emotions;
  /** \brief The head pose */
  HeadPoseDetection::Results _headPose;
  /** \brief The landmarks */
  std::vector<float> _landmarks;

  /** \brief True if age gender is enabled, false if not */
  bool _isAgeGenderEnabled;
  /** \brief True if emotions is enabled, false if not */
  bool _isEmotionsEnabled;
  /** \brief True if head pose is enabled, false if not */
  bool _isHeadPoseEnabled;
  /** \brief True if landmarks is enabled, false if not */
  bool _isLandmarksEnabled;
};

// ----------------------------------- Utils -----------------------------------------------------------------


/**********************************************************************************************//**
 * \fn  float calcIoU(cv::Rect& src, cv::Rect& dst);
 *
 * \brief Calculates the i/o u
 *
 * \author  Delmiro Paes
 *
 * \param [in,out]  src Source for the.
 * \param [in,out]  dst Destination for the.
 *
 * \returns The calculated i/o u.
 **************************************************************************************************/
float calcIoU(cv::Rect& src, cv::Rect& dst);


/**********************************************************************************************//**
 * \fn  float calcMean(const cv::Mat& src);
 *
 * \brief Finds the mean of the given arguments
 *
 * \author  Delmiro Paes
 *
 * \param src Source for the.
 *
 * \returns The calculated mean.
 **************************************************************************************************/
float calcMean(const cv::Mat& src);
Face::Ptr matchFace(cv::Rect rect, std::list<Face::Ptr>& faces);
