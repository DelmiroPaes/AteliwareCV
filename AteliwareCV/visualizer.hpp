//
// Copyright (C) 2019 Ateliware
//

/************************************************************************************************
 * \file  visualizer.hpp.
 *
 * \brief Implements the visualizer class
 **************************************************************************************************/

#pragma once

#include <memory>
#include <string>
#include <list>
#include <vector>
#include <map>
#include <opencv2/opencv.hpp>

#include "face.hpp"

 // -------------------------Generic routines for visualization of detection results-------------------------------------------------


 /**********************************************************************************************//**
  * \class EmotionBarVisualizer
  *
  * \brief Drawing a bar of emotions
  *
  * \author  Delmiro Paes
  **************************************************************************************************/
class EmotionBarVisualizer
{
public:
  /** \brief The pointer */
  using Ptr = std::shared_ptr<EmotionBarVisualizer>;


  /**********************************************************************************************//**
   * \fn  explicit EmotionBarVisualizer::EmotionBarVisualizer(std::vector<std::string> const& emotionNames, cv::Size size = cv::Size(300, 140), cv::Size padding = cv::Size(10, 10), double opacity = 0.6, double textScale = 1, int textThickness = 1);
   *
   * \brief Constructor
   *
   * \author  Delmiro Paes
   *
   * \param emotionNames  List of names of the emotions.
   * \param size          (Optional) The size.
   * \param padding       (Optional) The padding.
   * \param opacity       (Optional) The opacity.
   * \param textScale     (Optional) The text scale.
   * \param textThickness (Optional) The text thickness.
   **************************************************************************************************/
  explicit EmotionBarVisualizer(std::vector<std::string> const& emotionNames, cv::Size size = cv::Size(300, 140), cv::Size padding = cv::Size(10, 10),
    double opacity = 0.6, double textScale = 1, int textThickness = 1);


  /**********************************************************************************************//**
   * \fn  void EmotionBarVisualizer::draw(cv::Mat& img, std::map<std::string, float> emotions, cv::Point org, cv::Scalar fgcolor, cv::Scalar bgcolor);
   *
   * \brief Draws
   *
   * \author  Delmiro Paes
   *
   * \param [in,out]  img       The image.
   * \param           emotions  The emotions.
   * \param           org       The organisation.
   * \param           fgcolor   The fgcolor.
   * \param           bgcolor   The bgcolor.
   **************************************************************************************************/
  void draw(cv::Mat& img, std::map<std::string, float> emotions, cv::Point org, cv::Scalar fgcolor, cv::Scalar bgcolor);


  /**********************************************************************************************//**
   * \fn  cv::Size EmotionBarVisualizer::getSize();
   *
   * \brief Gets the size
   *
   * \author  Delmiro Paes
   *
   * \returns The size.
   **************************************************************************************************/
  cv::Size getSize();
private:
  /** \brief List of names of the emotions */
  std::vector<std::string> emotionNames;
  /** \brief The size */
  cv::Size size;
  /** \brief The padding */
  cv::Size padding;
  /** \brief Size of the text */
  cv::Size textSize;
  /** \brief The text baseline */
  int textBaseline;
  /** \brief The ystep */
  int ystep;
  /** \brief The opacity */
  double opacity;
  /** \brief The text scale */
  double textScale;
  /** \brief The text thickness */
  int textThickness;
  /** \brief The internal padding */
  int internalPadding;
};


/**********************************************************************************************//**
 * \class PhotoFrameVisualizer
 *
 * \brief Drawing a photo frame around detected face
 *
 * \author  Delmiro Paes
 **************************************************************************************************/
class PhotoFrameVisualizer
{
public:
  /** \brief The pointer */
  using Ptr = std::shared_ptr<PhotoFrameVisualizer>;


  /**********************************************************************************************//**
   * \fn  explicit PhotoFrameVisualizer::PhotoFrameVisualizer(int bbThickness = 1, int photoFrameThickness = 2, float photoFrameLength = 0.1);
   *
   * \brief Constructor
   *
   * \author  Delmiro Paes
   *
   * \param bbThickness         (Optional) The bb thickness.
   * \param photoFrameThickness (Optional) The photo frame thickness.
   * \param photoFrameLength    (Optional) Length of the photo frame.
   **************************************************************************************************/
  explicit PhotoFrameVisualizer(int bbThickness = 1, int photoFrameThickness = 2, float photoFrameLength = 0.1);


  /**********************************************************************************************//**
   * \fn  void PhotoFrameVisualizer::draw(cv::Mat& img, cv::Rect& bb, cv::Scalar color);
   *
   * \brief Draws
   *
   * \author  Delmiro Paes
   *
   * \param [in,out]  img   The image.
   * \param [in,out]  bb    The bb.
   * \param           color The color.
   **************************************************************************************************/
  void draw(cv::Mat& img, cv::Rect& bb, cv::Scalar color);

private:
  /** \brief The bb thickness */
  int bbThickness;
  /** \brief The photo frame thickness */
  int photoFrameThickness;
  /** \brief Length of the photo frame */
  float photoFrameLength;
};


/**********************************************************************************************//**
 * \class HeadPoseVisualizer
 *
 * \brief Drawing the position of the head
 *
 * \author  Delmiro Paes
 **************************************************************************************************/
class HeadPoseVisualizer
{
public:
  /** \brief The pointer */
  using Ptr = std::shared_ptr<HeadPoseVisualizer>;


  /**********************************************************************************************//**
   * \fn  explicit HeadPoseVisualizer::HeadPoseVisualizer(float scale = 50, cv::Scalar xAxisColor = cv::Scalar(0, 0, 255), cv::Scalar yAxisColor = cv::Scalar(0, 255, 0), cv::Scalar zAxisColor = cv::Scalar(255, 0, 0), int axisThickness = 2);
   *
   * \brief Constructor
   *
   * \author  Delmiro Paes
   *
   * \param scale         (Optional) The scale.
   * \param xAxisColor    (Optional) The axis color.
   * \param yAxisColor    (Optional) The axis color.
   * \param zAxisColor    (Optional) The axis color.
   * \param axisThickness (Optional) The axis thickness.
   **************************************************************************************************/
  explicit HeadPoseVisualizer(float scale = 50,
    cv::Scalar xAxisColor = cv::Scalar(0, 0, 255),
    cv::Scalar yAxisColor = cv::Scalar(0, 255, 0),
    cv::Scalar zAxisColor = cv::Scalar(255, 0, 0),
    int axisThickness = 2);


  /**********************************************************************************************//**
   * \fn  void HeadPoseVisualizer::draw(cv::Mat& frame, cv::Point3f cpoint, HeadPoseDetection::Results headPose);
   *
   * \brief Draws
   *
   * \author  Delmiro Paes
   *
   * \param [in,out]  frame     The frame.
   * \param           cpoint    The cpoint.
   * \param           headPose  The head pose.
   **************************************************************************************************/
  void draw(cv::Mat& frame, cv::Point3f cpoint, HeadPoseDetection::Results headPose);

private:

  /**********************************************************************************************//**
   * \fn  void HeadPoseVisualizer::buildCameraMatrix(cv::Mat& cameraMatrix, int cx, int cy, float focalLength);
   *
   * \brief Builds camera matrix
   *
   * \author  Delmiro Paes
   *
   * \param [in,out]  cameraMatrix  The camera matrix.
   * \param           cx            The cx.
   * \param           cy            The cy.
   * \param           focalLength   Length of the focal.
   **************************************************************************************************/
  void buildCameraMatrix(cv::Mat& cameraMatrix, int cx, int cy, float focalLength);

private:
  /** \brief The axis color */
  cv::Scalar xAxisColor;
  /** \brief The axis color */
  cv::Scalar yAxisColor;
  /** \brief The axis color */
  cv::Scalar zAxisColor;
  /** \brief The axis thickness */
  int axisThickness;
  /** \brief The scale */
  float scale;
};


/**********************************************************************************************//**
 * \class Visualizer
 *
 * \brief Drawing detected faces on the frame
 *
 * \author  Delmiro Paes
 **************************************************************************************************/
class Visualizer
{
public:
  /** \brief The pointer */
  using Ptr = std::shared_ptr<Visualizer>;


  /**********************************************************************************************//**
   * \enum  AnchorType
   *
   * \brief Values that represent anchor types
   **************************************************************************************************/
  enum AnchorType
  {
    TL = 0,
    TR,
    BL,
    BR
  };


  /**********************************************************************************************//**
   * \struct  DrawParams
   *
   * \brief A draw parameters.
   *
   * \author  Delmiro Paes
   **************************************************************************************************/
  struct DrawParams
  {
    cv::Point cell;
    /** \brief The bar anchor */
    AnchorType barAnchor;
    /** \brief The rectangle anchor */
    AnchorType rectAnchor;
    /** \brief Zero-based index of the frame */
    size_t frameIdx;
  };


  /**********************************************************************************************//**
   * \fn  explicit Visualizer::Visualizer(cv::Size const& imgSize, int leftPadding = 10, int rightPadding = 10, int topPadding = 75, int bottomPadding = 10);
   *
   * \brief Constructor
   *
   * \author  Delmiro Paes
   *
   * \param imgSize       Size of the image.
   * \param leftPadding   (Optional) The left padding.
   * \param rightPadding  (Optional) The right padding.
   * \param topPadding    (Optional) The top padding.
   * \param bottomPadding (Optional) The bottom padding.
   **************************************************************************************************/
  explicit Visualizer(cv::Size const& imgSize, int leftPadding = 10, int rightPadding = 10, int topPadding = 75, int bottomPadding = 10);


  /**********************************************************************************************//**
   * \fn  void Visualizer::enableEmotionBar(std::vector<std::string> const& emotionNames);
   *
   * \brief Enables the emotion bar
   *
   * \author  Delmiro Paes
   *
   * \param emotionNames  List of names of the emotions.
   **************************************************************************************************/
  void enableEmotionBar(std::vector<std::string> const& emotionNames);


  /**********************************************************************************************//**
   * \fn  void Visualizer::draw(cv::Mat img, std::list<Face::Ptr> faces);
   *
   * \brief Draws
   *
   * \author  Delmiro Paes
   *
   * \param img   The image.
   * \param faces The faces.
   **************************************************************************************************/
  void draw(cv::Mat img, std::list<Face::Ptr> faces);

private:

  /**********************************************************************************************//**
   * \fn  void Visualizer::drawFace(cv::Mat& img, Face::Ptr f, bool drawEmotionBar);
   *
   * \brief Draw face
   *
   * \author  Delmiro Paes
   *
   * \param [in,out]  img             The image.
   * \param           f               A Face::Ptr to process.
   * \param           drawEmotionBar  True to draw emotion bar.
   **************************************************************************************************/
  void drawFace(cv::Mat& img, Face::Ptr f, bool drawEmotionBar);


  /**********************************************************************************************//**
   * \fn  cv::Point Visualizer::findCellForEmotionBar();
   *
   * \brief Searches for the first cell for emotion bar
   *
   * \author  Delmiro Paes
   *
   * \returns The found cell for emotion bar.
   **************************************************************************************************/
  cv::Point findCellForEmotionBar();

  /** \brief Options for controlling the draw */
  std::map<size_t, DrawParams> drawParams;
  /** \brief The emotion visualizer */
  EmotionBarVisualizer::Ptr emotionVisualizer;
  /** \brief The photo frame visualizer */
  PhotoFrameVisualizer::Ptr photoFrameVisualizer;
  /** \brief The head pose visualizer */
  HeadPoseVisualizer::Ptr headPoseVisualizer;

  /** \brief The draw map */
  cv::Mat drawMap;
  /** \brief The nxcells */
  int nxcells;
  /** \brief The nycells */
  int nycells;
  /** \brief The xstep */
  int xstep;
  /** \brief The ystep */
  int ystep;

  /** \brief Size of the image */
  cv::Size imgSize;
  /** \brief The left padding */
  int leftPadding;
  /** \brief The right padding */
  int rightPadding;
  /** \brief The top padding */
  int topPadding;
  /** \brief The bottom padding */
  int bottomPadding;
  /** \brief /** \brief Size of the emotion bar */
  cv::Size emotionBarSize;
  size_t frameCounter;
};
