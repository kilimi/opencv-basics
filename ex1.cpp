#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/operations.hpp>

#include <iostream>
#include <stdio.h>

using namespace cv;
using namespace std;


cv::Mat applyLookUpTable(const cv::Mat& image, const cv::Mat& lookup)
{
	cv::Mat result;
	cv::LUT(image,lookup,result); 	// apply lookup table
	return result;
}
cv::Mat stretchImage(const cv::Mat &image, cv::MatND &hist, int minValue=0)
{
	int imin= 0;
	for( ; imin < 256; imin++ ) //find the lowest intensity
	{
		if (hist.at<float>(imin) > minValue)
			break;
	}
	// find right extremity of the histogram
	int imax= 255;
	for( ; imax >= 0; imax-- ) {
		if (hist.at<float>(imax) > minValue)
			break;
	}
	// create lookup table
	cv::Mat lookup(1, 256,	CV_8U);
	// build lookup table
	for (int i=0; i<256; i++)
	{
		// stretch between imin and imax
		if (i < imin) lookup.at<uchar>(i)= 0;
		else if (i > imax) lookup.at<uchar>(i)= 255;
		// linear mapping
		else lookup.at<uchar>(i)= static_cast<uchar>(255.0*(i-imin)/(imax-imin)+0.5);
	}
	cv::Mat result = applyLookUpTable(image,lookup);
	return result;
}
cv::Mat filtering(cv::Mat &img)
{
	cv::Mat imgCopy;
	img.copyTo(imgCopy);
	int r = imgCopy.rows;
	int c = imgCopy.cols;

	cv::copyMakeBorder(imgCopy, imgCopy, 1, 1, 1, 1, cv::BORDER_REPLICATE);
	for (int j=1; j<r-1; j++)
	{
		uchar* rowBefore = imgCopy.ptr<uchar>(j - 1);
		uchar* row = imgCopy.ptr<uchar>(j);
		uchar* rowAfter = imgCopy.ptr<uchar>(j + 1);

		for (int i=1; i<c-1; i++)
		{
			cv::Mat arr(3,3,CV_8U,cv::Scalar(0));
			Mat res;
			arr.at<uchar>(1,0) = row[i-1];
			arr.at<uchar>(1,1) = row[i];
			arr.at<uchar>(1,2) = row[i+1];
			arr.at<uchar>(0,0) = rowBefore[i-1];
			arr.at<uchar>(0,1) = rowBefore[i];
			arr.at<uchar>(0,2) = rowBefore[i+1];
			arr.at<uchar>(2,0) = rowAfter[i-1];
			arr.at<uchar>(2,1) = rowAfter[i];
			arr.at<uchar>(2,2) = rowAfter[i+1];

			cv::Mat combined = arr.reshape(1, 1);
			cv::sort(combined, res, CV_SORT_EVERY_ROW|CV_SORT_ASCENDING);
			float mean = (res.at<uchar>(3) + res.at<uchar>(4)  + res.at<uchar>(5) )/3;
			row[i] =  mean;
		}
	}
	return imgCopy;
}
cv::Mat intensityIncrease(const cv::Mat &img, int amount)
{
	cv::Mat imgCopy;
	img.copyTo(imgCopy);
	int r= imgCopy.rows;
	int c = imgCopy.cols;
	// for all pixels
	for (int j=0; j<r; j++)
	{
		// pointer to first column of line j
		uchar* data= imgCopy.ptr<uchar>(j);
		for (int i=0; i<c; i++)
		{
			data[i] = cv::saturate_cast<uchar>(data[i] + amount);
		} //end of line
	}
	return imgCopy;
}

cv::MatND getHistogram(const cv::Mat &image)
{
	int numOfBins[1];
	float minMaxValue[2];
	const float* ranges[1];
	int channels[1];
	numOfBins[0] = 256;
	minMaxValue[0] = 0.0;
	minMaxValue[1] = 255.0;
	ranges[0] = minMaxValue;
	channels[0] = 0;
	int dimentions = 1;
	int numOfIm = 1;
	cv::MatND hist;
	cv::calcHist(&image, numOfIm, channels, cv::Mat(), hist, dimentions, numOfBins, ranges);
	return hist;
}
cv::Mat getHistogramImage(const cv::Mat &image)
{
	int numOfBins[1];
	numOfBins[0] = 256;
	cv::MatND hist= getHistogram(image);
	// Get min and max bin values
	double maxVal=0;
	double minVal=0;
	cv::minMaxLoc(hist, &minVal, &maxVal, 0, 0);

	cv::Mat histImg(numOfBins[0], numOfBins[0], CV_8U,cv::Scalar(255));

	int hpt = static_cast<int>(0.9*numOfBins[0]); // set highest point at 90% of nbins

	for( int h = 0; h < numOfBins[0]; h++ ) // Draw a vertical line for each bin
	{
		float binVal = hist.at<float>(h);
		int intensity = static_cast<int>(binVal*hpt/maxVal);
		cv::line(histImg,cv::Point(h,numOfBins[0]), cv::Point(h,numOfBins[0]-intensity), cv::Scalar::all(0));
	}
	return histImg;
}


int main(int argc, char* argv[])
{
	cv::Mat img = cv::imread("/home/lilith/Desktop/lena.bmp");
	cv::Mat res, intensityTr;
	cv::cvtColor(img, res, CV_BGR2GRAY);

	cv::namedWindow("Picture in grayscale");
	cv::imshow("Picture in grayscale", res);
	cv::waitKey(0);

	//----------Exercise 1
	cv::namedWindow("Changed Intensity");
	cv::imshow("Changed Intensity", intensityIncrease(res, 50));
	cv::waitKey(0);

	//-----------Exercise 2
	cv::Mat hist = getHistogramImage(res);
	cv::namedWindow("Histogram");
	cv::imshow("Histogram", hist);
	cv::waitKey(0);

	// look up table
	cv::MatND h = getHistogram(res);
	cv::Mat stretchedImage = stretchImage(res, h, 256);
	cv::Mat hist2 = getHistogramImage(stretchedImage);
	cv::namedWindow("Histogram after image stretching");
	cv::imshow("Histogram after image stretching", hist2);

	cv::namedWindow("After stretching");
	cv::imshow("After stretching", stretchedImage);

	cv::waitKey(0);

	//----Equalizing the image histogram
	cv::Mat eqHist;
	cv::equalizeHist(res, eqHist);
	cv::namedWindow("Equalizing the image histogram");
	cv::imshow("Equalizing the image histogram", eqHist);

	cv::waitKey(0);

	//-------Exercise 3
	cv::Mat afterKernel;

	Mat kernel = (Mat_<float>(3,3) <<
			0,  1, 0,
			1, -4, 1,
			0,  1, 0);

//	cout << kernel << endl;
	int ddepth = -1; //the same as source image
	cv::filter2D(res, afterKernel, ddepth, kernel, Point(-1,-1), 0.0, BORDER_REPLICATE);

	cv::namedWindow("After Kernel");
	cv::imshow("After Kernel", afterKernel);

	cv::namedWindow("Custom filtering");
	cv::imshow("Custom filtering", filtering(res));
	cv::waitKey(0);

	return 0;
}


