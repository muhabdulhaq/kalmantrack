/******************************************
Muhammad Abdul Haq
PENS - CE16
2210161005
 ******************************************/
#include <cv.h>
#include <highgui.h>
#include <opencv/cxcore.h>
#include <fstream>
#include <iostream>
#include <cmath>
#include <stdio.h>
#include <fstream>
#include <vector>
#include <cstring>
#include <ctime>
#include <limits.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/stat.h>
#include <pthread.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/video.hpp>
#include <stdlib.h>
using namespace std;
using namespace cv;
int Hmin=0, Hmax=62, Smin=100, Smax=255, Vmin=80, Vmax=255, E=0, D=0;
void trackBar(){
    cvCreateTrackbar("H/Y MIN", "Result", &Hmin, 255, 0);
    cvCreateTrackbar("S/U MIN", "Result", &Smin, 255, 0);
    cvCreateTrackbar("V MIN", "Result", &Vmin, 255, 0);
    cvCreateTrackbar("H/Y MAX", "Result", &Hmax, 255, 0);
    cvCreateTrackbar("S/U MAX", "Result", &Smax, 255, 0);
    cvCreateTrackbar("V MAX", "Result", &Vmax, 255, 0);
    cvCreateTrackbar("E", "Result", &E, 100, 0);
    cvCreateTrackbar("D", "Result", &D, 100, 0);
}
cv::Rect predRect;
cv::Point kfc;
int main()
{
    cv::Mat frame;
cvNamedWindow("Result");
    int stateSize = 6;
    int measSize = 4;
    int contrSize = 0;
    unsigned int type = CV_32F;
    cv::KalmanFilter kf(stateSize, measSize, contrSize, type);
    cv::Mat state(stateSize, 1, type);  
    cv::Mat meas(measSize, 1, type);   
    cv::setIdentity(kf.transitionMatrix);
    kf.measurementMatrix = cv::Mat::zeros(measSize, stateSize, type);
    kf.measurementMatrix.at<float>(0) = 1.0f;
    kf.measurementMatrix.at<float>(7) = 1.0f;
    kf.measurementMatrix.at<float>(16) = 1.0f;
    kf.measurementMatrix.at<float>(23) = 1.0f;
    kf.processNoiseCov.at<float>(0) = 1e-2;
    kf.processNoiseCov.at<float>(7) = 1e-2;
    kf.processNoiseCov.at<float>(14) = 5.0f;
    kf.processNoiseCov.at<float>(21) = 5.0f;
    kf.processNoiseCov.at<float>(28) = 1e-2;
    kf.processNoiseCov.at<float>(35) = 1e-2;
    cv::setIdentity(kf.measurementNoiseCov, cv::Scalar(1e-1));
    int idx = 1;
    cv::VideoCapture cap(0);
    if (!cap.open(idx))
    {
        cout << "Webcam not connected.\n" << "Please verify\n";
        return EXIT_FAILURE;
    }

    cap.set(CV_CAP_PROP_FRAME_WIDTH, 1024);
    cap.set(CV_CAP_PROP_FRAME_HEIGHT, 768);
    cout << "\nHit 'q' to exit...\n";
    char ch = 0;
    double ticks = 0;
    bool found = false;
    int notFoundCount = 0;
    while (ch != 'q' && ch != 'Q'){
        trackBar();
        double precTick = ticks;
        ticks = (double) cv::getTickCount();
        double dT = (ticks - precTick) / cv::getTickFrequency();
        cap >> frame;
        cv::Mat res;
        frame.copyTo( res );
        if (found)
        {
            kf.transitionMatrix.at<float>(2) = dT;
            kf.transitionMatrix.at<float>(9) = dT;
            state = kf.predict();
            predRect.width = state.at<float>(4);
            predRect.height = state.at<float>(5);
            predRect.x = state.at<float>(0) - predRect.width / 2;
            predRect.y = state.at<float>(1) - predRect.height / 2;
            cv::Point center;
            center.x = state.at<float>(0);
            center.y = state.at<float>(1);
            kfc.x = state.at<float>(0) ;
            kfc.y = state.at<float>(1) ;
            cv::circle(res, kfc, 5, CV_RGB(255,0,0), -1);
            cv::rectangle(res, predRect, CV_RGB(255,0,0), 2);
        }
        cv::Mat blur;
        cv::GaussianBlur(frame, blur, cv::Size(5, 5), 3.0, 3.0);
        cv::Mat frmHsv;
        cv::cvtColor(blur, frmHsv, CV_BGR2HSV);
        cv::Mat rangeRes = cv::Mat::zeros(frame.size(), CV_8UC1);
        cv::inRange(frmHsv, cv::Scalar(Hmin/2, Smin, Vmin),               cv::Scalar(Hmax/2, Smax, Vmax), rangeRes);
        cv::erode(rangeRes, rangeRes, cv::Mat(), cv::Point(-1, -1), 2);
        cv::dilate(rangeRes, rangeRes, cv::Mat(), cv::Point(-1, -1), 2);
        cv::imshow("Threshold", rangeRes);
        vector<vector<cv::Point> > contours;
        cv::findContours(rangeRes, contours, CV_RETR_EXTERNAL,
                         CV_CHAIN_APPROX_NONE);
        vector<vector<cv::Point> > balls;
        vector<cv::Rect> ballsBox;
        for (size_t i = 0; i < contours.size(); i++)
        {
            cv::Rect bBox;
            bBox = cv::boundingRect(contours[i]);

            float ratio = (float) bBox.width / (float) bBox.height;
            if (ratio > 1.0f)
                ratio = 1.0f / ratio;

            // Searching for a bBox almost square
            if (ratio > 0.75 && bBox.area() >= 400)
            {
                balls.push_back(contours[i]);
                ballsBox.push_back(bBox);
            }
        }
        // <<<<< Filtering

        cout << "Balls found:" << ballsBox.size() << endl;

        // >>>>> Detection result
        for (size_t i = 0; i < balls.size(); i++)
        {
            cv::drawContours(res, balls, i, CV_RGB(20,150,20), 1);
            cv::rectangle(res, ballsBox[i], CV_RGB(0,255,0), 2);

            cv::Point center;
            center.x = ballsBox[i].x + ballsBox[i].width / 2;
            center.y = ballsBox[i].y + ballsBox[i].height / 2;
            cv::circle(res, center, 5, CV_RGB(20,150,20), -1);

            //px1 512 & py1 768 --> garis px2 center.x & py2 center.y --> bola | ukuran frame 1024 * 768
            int angle = (double)atan2( center.y - 768, center.x - 512)* 180 / CV_PI;
            printf("\nXB : %d XKF : %d X : %d YB : %d YKF : %d Y : %d Teta B : %d\n", center.x, kfc.x, abs(center.x-kfc.x), center.y, kfc.y, abs(center.y-kfc.y), angle);
            line(res, Point(512, 768), Point(center.x,center.y), cv::Scalar(0,0,255), 1);

            stringstream sstr;
            sstr << "(" << center.x << "," << center.y << ")";
            cv::putText(res, sstr.str(),
                        cv::Point(center.x + 3, center.y - 3),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, CV_RGB(20,150,20), 2);
        }
        // <<<<< Detection result

        // >>>>> Kalman Update
        if (balls.size() == 0)
        {
            notFoundCount++;
            cout << "notFoundCount:" << notFoundCount << endl;
            if( notFoundCount >= 100 )
            {
                found = false;
            }
            /*else
                kf.statePost = state;*/
        }
        else
        {
            notFoundCount = 0;

            meas.at<float>(0) = ballsBox[0].x + ballsBox[0].width / 2;
            meas.at<float>(1) = ballsBox[0].y + ballsBox[0].height / 2;
            meas.at<float>(2) = (float)ballsBox[0].width;
            meas.at<float>(3) = (float)ballsBox[0].height;

            if (!found) // First detection!
            {
                // >>>> Initialization
                kf.errorCovPre.at<float>(0) = 1; // px
                kf.errorCovPre.at<float>(7) = 1; // px
                kf.errorCovPre.at<float>(14) = 1;
                kf.errorCovPre.at<float>(21) = 1;
                kf.errorCovPre.at<float>(28) = 1; // px
                kf.errorCovPre.at<float>(35) = 1; // px

                state.at<float>(0) = meas.at<float>(0);
                state.at<float>(1) = meas.at<float>(1);
                state.at<float>(2) = 0;
                state.at<float>(3) = 0;
                state.at<float>(4) = meas.at<float>(2);
                state.at<float>(5) = meas.at<float>(3);
                // <<<< Initialization

                kf.statePost = state;
                
                found = true;
            }
            else
                kf.correct(meas); // Kalman Correction

           // cout << "Measure matrix:" << endl << meas << endl;
        }
        // <<<<< Kalman Update

        // Final result
        cv::imshow("Tracking", res);

        // User key
        ch = cv::waitKey(1);
    }
    // <<<<< Main loop

    return EXIT_SUCCESS;
}
