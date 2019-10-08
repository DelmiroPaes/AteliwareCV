//
// Copyright (C) 2019 Ateliware
//

/************************************************************************************************
 * \file  face.cpp.
 *
 * \brief Implements the face class
 **************************************************************************************************/

#pragma warning( disable : 4251 )
#pragma warning( disable : 4267 )
#pragma warning( disable : 4275 )

#include <string>
#include <map>
#include <utility>
#include <list>
#include <vector>

#include "face.hpp"


 /**********************************************************************************************//**
  * \fn  Face::Face(size_t id, cv::Rect& location)
  *
  * \brief Constructor
  *
  * \author  Delmiro Paes
  *
  * \param           id        The identifier.
  * \param [in,out]  location  The location.
  **************************************************************************************************/
Face::Face(size_t id, cv::Rect& location) :
  _location(location), _intensity_mean(0.f), _id(id), _age(-1),
  _maleScore(0), _femaleScore(0), _headPose({ 0.f, 0.f, 0.f }),
  _isAgeGenderEnabled(false), _isEmotionsEnabled(false), _isHeadPoseEnabled(false), _isLandmarksEnabled(false)
{
}


/**********************************************************************************************//**
 * \fn  void Face::updateAge(float value)
 *
 * \brief Updates the age described by value
 *
 * \author  Delmiro Paes
 *
 * \param value The value.
 **************************************************************************************************/
void Face::updateAge(float value) 
{
  _age = (_age == -1) ? value : 0.95f * _age + 0.05f * value;
}


/**********************************************************************************************//**
 * \fn  void Face::updateGender(float value)
 *
 * \brief Updates the gender described by value
 *
 * \author  Delmiro Paes
 *
 * \param value The value.
 **************************************************************************************************/
void Face::updateGender(float value) 
{
  if (value < 0)
    return;

  if (value > 0.5) 
  {
    _maleScore += value - 0.5f;
  }
  else 
  {
    _femaleScore += 0.5f - value;
  }
}


/**********************************************************************************************//**
 * \fn  void Face::updateEmotions(std::map<std::string, float> values)
 *
 * \brief Updates the emotions described by values
 *
 * \author  Delmiro Paes
 *
 * \param values  The values.
 **************************************************************************************************/
void Face::updateEmotions(std::map<std::string, float> values) 
{
  for (auto& kv : values) 
  {
    if (_emotions.find(kv.first) == _emotions.end()) 
    {
      _emotions[kv.first] = kv.second;
    }
    else 
    {
      _emotions[kv.first] = 0.9f * _emotions[kv.first] + 0.1f * kv.second;
    }
  }
}


/**********************************************************************************************//**
 * \fn  void Face::updateHeadPose(HeadPoseDetection::Results values)
 *
 * \brief Updates the head pose described by values
 *
 * \author  Delmiro Paes
 *
 * \param values  The values.
 **************************************************************************************************/
void Face::updateHeadPose(HeadPoseDetection::Results values) 
{
  _headPose = values;
}


/**********************************************************************************************//**
 * \fn  void Face::updateLandmarks(std::vector<float> values)
 *
 * \brief Updates the landmarks described by values
 *
 * \author  Delmiro Paes
 *
 * \param values  The values.
 **************************************************************************************************/
void Face::updateLandmarks(std::vector<float> values) 
{
  _landmarks = std::move(values);
}


/**********************************************************************************************//**
 * \fn  int Face::getAge()
 *
 * \brief Gets the age
 *
 * \author  Delmiro Paes
 *
 * \returns The age.
 **************************************************************************************************/
int Face::getAge() 
{
  return static_cast<int>(std::floor(_age + 0.5f));
}


/**********************************************************************************************//**
 * \fn  bool Face::isMale()
 *
 * \brief Query if this object is male
 *
 * \author  Delmiro Paes
 *
 * \returns True if male, false if not.
 **************************************************************************************************/
bool Face::isMale() 
{
  return _maleScore > _femaleScore;
}


/**********************************************************************************************//**
 * \fn  std::map<std::string, float> Face::getEmotions()
 *
 * \brief Gets the emotions
 *
 * \author  Delmiro Paes
 *
 * \returns The emotions.
 **************************************************************************************************/
std::map<std::string, float> Face::getEmotions() 
{
  return _emotions;
}


/**********************************************************************************************//**
 * \fn  std::pair<std::string, float> Face::getMainEmotion()
 *
 * \brief Gets main emotion
 *
 * \author  Delmiro Paes
 *
 * \returns The main emotion.
 **************************************************************************************************/
std::pair<std::string, float> Face::getMainEmotion() 
{
  auto x = std::max_element(_emotions.begin(), _emotions.end(), [](const std::pair<std::string, float>& p1, const std::pair<std::string, float>& p2) 
  {
    return p1.second < p2.second; 
  });

  return std::make_pair(x->first, x->second);
}


/**********************************************************************************************//**
 * \fn  HeadPoseDetection::Results Face::getHeadPose()
 *
 * \brief Gets head pose
 *
 * \author  Delmiro Paes
 *
 * \returns The head pose.
 **************************************************************************************************/
HeadPoseDetection::Results Face::getHeadPose() 
{
  return _headPose;
}


/**********************************************************************************************//**
 * \fn  const std::vector<float>& Face::getLandmarks()
 *
 * \brief Gets the landmarks
 *
 * \author  Delmiro Paes
 *
 * \returns The landmarks.
 **************************************************************************************************/
const std::vector<float>& Face::getLandmarks() 
{
  return _landmarks;
}


/**********************************************************************************************//**
 * \fn  size_t Face::getId()
 *
 * \brief Gets the identifier
 *
 * \author  Delmiro Paes
 *
 * \returns The identifier.
 **************************************************************************************************/
size_t Face::getId() 
{
  return _id;
}


/**********************************************************************************************//**
 * \fn  void Face::ageGenderEnable(bool value)
 *
 * \brief Age gender enable
 *
 * \author  Delmiro Paes
 *
 * \param value True to value.
 **************************************************************************************************/
void Face::ageGenderEnable(bool value) 
{
  _isAgeGenderEnabled = value;
}


/**********************************************************************************************//**
 * \fn  void Face::emotionsEnable(bool value)
 *
 * \brief Emotions enable
 *
 * \author  Delmiro Paes
 *
 * \param value True to value.
 **************************************************************************************************/
void Face::emotionsEnable(bool value) 
{
  _isEmotionsEnabled = value;
}


/**********************************************************************************************//**
 * \fn  void Face::headPoseEnable(bool value)
 *
 * \brief Head pose enable
 *
 * \author  Delmiro Paes
 *
 * \param value True to value.
 **************************************************************************************************/
void Face::headPoseEnable(bool value) 
{
  _isHeadPoseEnabled = value;
}


/**********************************************************************************************//**
 * \fn  void Face::landmarksEnable(bool value)
 *
 * \brief Landmarks enable
 *
 * \author  Delmiro Paes
 *
 * \param value True to value.
 **************************************************************************************************/
void Face::landmarksEnable(bool value) 
{
  _isLandmarksEnabled = value;
}


/**********************************************************************************************//**
 * \fn  bool Face::isAgeGenderEnabled()
 *
 * \brief Queries if the age gender is enabled
 *
 * \author  Delmiro Paes
 *
 * \returns True if the age gender is enabled, false if not.
 **************************************************************************************************/
bool Face::isAgeGenderEnabled() 
{
  return _isAgeGenderEnabled;
}


/**********************************************************************************************//**
 * \fn  bool Face::isEmotionsEnabled()
 *
 * \brief Queries if the emotions is enabled
 *
 * \author  Delmiro Paes
 *
 * \returns True if the emotions is enabled, false if not.
 **************************************************************************************************/
bool Face::isEmotionsEnabled() 
{
  return _isEmotionsEnabled;
}


/**********************************************************************************************//**
 * \fn  bool Face::isHeadPoseEnabled()
 *
 * \brief Queries if the head pose is enabled
 *
 * \author  Delmiro Paes
 *
 * \returns True if the head pose is enabled, false if not.
 **************************************************************************************************/
bool Face::isHeadPoseEnabled() 
{
  return _isHeadPoseEnabled;
}


/**********************************************************************************************//**
 * \fn  bool Face::isLandmarksEnabled()
 *
 * \brief Queries if the landmarks is enabled
 *
 * \author  Delmiro Paes
 *
 * \returns True if the landmarks is enabled, false if not.
 **************************************************************************************************/
bool Face::isLandmarksEnabled() 
{
  return _isLandmarksEnabled;
}


/**********************************************************************************************//**
 * \fn  float calcIoU(cv::Rect& src, cv::Rect& dst)
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
float calcIoU(cv::Rect& src, cv::Rect& dst) 
{
  cv::Rect i = src & dst;
  cv::Rect u = src | dst;

  return static_cast<float>(i.area()) / static_cast<float>(u.area());
}


/**********************************************************************************************//**
 * \fn  float calcMean(const cv::Mat& src)
 *
 * \brief Finds the mean of the given arguments
 *
 * \author  Delmiro Paes
 *
 * \param src Source for the.
 *
 * \returns The calculated mean.
 **************************************************************************************************/
float calcMean(const cv::Mat& src) 
{
  cv::Mat tmp;
  cv::cvtColor(src, tmp, cv::COLOR_BGR2GRAY);
  cv::Scalar mean = cv::mean(tmp);

  return static_cast<float>(mean[0]);
}


/**********************************************************************************************//**
 * \fn  Face::Ptr matchFace(cv::Rect rect, std::list<Face::Ptr>& faces)
 *
 * \brief Match face
 *
 * \author  Delmiro Paes
 *
 * \param           rect  The rectangle.
 * \param [in,out]  faces The faces.
 *
 * \returns A Face::Ptr.
 **************************************************************************************************/
Face::Ptr matchFace(cv::Rect rect, std::list<Face::Ptr>& faces) 
{
  Face::Ptr face(nullptr);
  float maxIoU = 0.55f;
  for (auto&& f : faces) 
  {
    float iou = calcIoU(rect, f->_location);
    if (iou > maxIoU) 
    {
      face = f;
      maxIoU = iou;
    }
  }

  return face;
}
