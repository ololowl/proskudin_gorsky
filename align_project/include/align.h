#pragma once

#include "io.h"
	#include "matrix.h"

Image align(const Image &srcImage, bool isPostprocessing, std::string postprocessingType, double fraction, bool isMirror, 
            bool isInterp, bool isSubpixel, double subScale); // 	+ 5

Image sobel_x(Image src_image);

Image sobel_y(Image src_image);

Image unsharp(const Image &src_image); // 				+ 1 = 6

Image gray_world(const Image &src_image); // 				+ 1 = 7

Image resize(const Image &src_image, double scale); // 			+ 2 = 9

Image custom(Image src_image, Matrix<double> kernel);

Image autocontrast(const Image &src_image, double fraction); //		+ 1 = 10

Image gaussian(Image src_image, double sigma, int radius);

Image gaussian_separable(Image src_image, double sigma, int radius);
// mirror								+ 2 = 12
Image median(const Image &src_image, uint radius); // 			+ 1 = 13

Image median_linear(const Image &src_image, uint radius);

Image median_const(Image src_image, int radius);

Image canny(Image src_image, int threshold1, int threshold2);
