#include "align.h"
#include <string>
#include <algorithm>
#include <cfloat>
#include <vector>

#define EPS 10e-5

enum { RAD = 30, DIAM = 60 , MINLEN = 300, SCALE = 2 };
// enum { RAD = 100, DIAM = 200 };
int pyramid_radius = RAD;
using uint = unsigned int;

Matrix<uint> get_greyscale(const Image &img)
{
    Matrix<uint> m(img.n_rows, img.n_cols);
    for (uint i = 0; i < img.n_rows; ++i){
        for (uint j = 0; j < img.n_cols; ++j){
            m(i, j) = std::get<0>(img(i, j));
        }
    }
    return m;
}

double mse(const Matrix<uint> &img1, const Matrix<uint> &img2)
{
    const uint rows = img1.n_rows;
    const uint cols = img1.n_cols;
    
    unsigned long long sum = 0;
    for (uint i = 0; i < rows; ++i) {
        for (uint j = 0; j < cols; ++j) {
            uint tmp = img1(i, j) - img2(i, j);
            sum += tmp * tmp;
        }
    }
    return double(sum) / (rows * cols);
}

double cross_corr(const Matrix<uint> &img1, const Matrix<uint> &img2)
{
    const uint rows = img1.n_rows;
    const uint cols = img1.n_cols;

    unsigned long long sum = 0;
    for (uint i = 0; i < rows; ++i) {
        for (uint j = 0; j < cols; ++j) {
            sum += img1(i, j) * img2(i, j);
        }
    }
    return double(sum);
}

void split3 (const Matrix<uint> &src, Matrix<uint> &r, Matrix<uint> &g, Matrix<uint> &b)
{
    const uint rows = src.n_rows / 3;
    const uint cols = src.n_cols;
    b = src.submatrix(0, 0, rows, cols);
    g = src.submatrix(rows, 0, rows, cols);
    r = src.submatrix(rows * 2, 0, rows, cols);
}

double find_shift_mse(const Matrix<uint> &m1, const Matrix<uint> &m2, int &shift_r, int &shift_c)
{
    Matrix<uint> window1 = m1.submatrix(RAD, RAD, m1.n_rows - DIAM, m1.n_cols - DIAM);
    double metric_min = DBL_MAX;
    for (uint row = 0; row < DIAM; ++row) {
        for (uint col = 0; col < DIAM; ++col) {
            Matrix<uint> window2 = m2.submatrix(row, col, m2.n_rows - DIAM, m2.n_cols - DIAM);
            double t = mse(window1, window2);
            if (metric_min - t > EPS) {
                metric_min = t;
                shift_r = row - RAD;
                shift_c = col - RAD;
            }
        }
    }
    return metric_min;
}

double find_shift_cc(const Matrix<uint> &m1, const Matrix<uint> &m2, int &shift_r, int &shift_c)
{
    Matrix<uint> window1 = m1.submatrix(RAD, RAD, m1.n_rows - DIAM, m1.n_cols - DIAM);
    double metric_max = 0;
    for (uint row = 0; row < DIAM; ++row) {
        for (uint col = 0; col < DIAM; ++col) {
            Matrix<uint> window2 = m2.submatrix(row, col, m2.n_rows - DIAM, m2.n_cols - DIAM);
            double t = cross_corr(window1, window2);  
            if (t >= metric_max) {
                metric_max = t;
                shift_r = row - RAD;
                shift_c = col - RAD;
            }
        }
    }
    return metric_max;
}

void cut_before_align (Matrix<uint> &m1, Matrix<uint> &m2, Matrix<uint> &m3,
                int shiftRow12, int shiftCol12, int shiftRow13, int shiftCol13)
{
    uint rows = m1.n_rows;
    uint cols = m1.n_cols;
    

    uint startRow = std::max(std::max(0, shiftRow12), shiftRow13);
    uint startCol = std::max(std::max(0, shiftCol12), shiftCol13);
    uint endRow = std::min(std::min(rows, rows + shiftRow12), rows + shiftRow13);
    uint endCol = std::min(std::min(cols, cols + shiftCol12), cols + shiftCol13);
    m1 = m1.submatrix(startRow, startCol, endRow - startRow, endCol - startCol);

    startRow = std::max(std::max(0, -shiftRow12), -shiftRow12 + shiftRow13);
    startCol = std::max(std::max(0, -shiftCol12), -shiftCol12 + shiftCol13);
    endRow = std::min(std::min(rows, rows - shiftRow12), rows - shiftRow12 + shiftRow13);
    endCol = std::min(std::min(cols, cols - shiftCol12), cols - shiftCol12 + shiftCol13);
    m2 = m2.submatrix(startRow, startCol, endRow - startRow, endCol - startCol);
    
    startRow = std::max(std::max(0, -shiftRow13), -shiftRow13 + shiftRow12);
    startCol = std::max(std::max(0, -shiftCol13), -shiftCol13 + shiftCol12);
    endRow = std::min(std::min(rows, rows - shiftRow13), rows - shiftRow13 + shiftRow12);
    endCol = std::min(std::min(cols, cols - shiftCol13), cols - shiftCol13 + shiftCol12);
    m3 = m3.submatrix(startRow, startCol, endRow - startRow, endCol - startCol);    
}

uint bilinear_interp(const Matrix<uint> &img, double x, double y,
                                              double x1, double y1, 
                                              double x2, double y2)
{
    uint r11, r12, r21, r22;
    r11 = img(x1, y1);
    r12 = img(x1, y2);
    r21 = img(x2, y1);
    r22 = img(x2, y2);
    uint r = r11 * (x2 - x) * (y2 - y) + r21 * (x - x1) * (y2 - y) +
             r12 * (x2 - x) * (y - y1) + r22 * (x - x1) * (y - y1);
    return r;
}

std::tuple<uint, uint, uint> bilinear_interp(const Image &img, double x, double y, 
                                                               double x1, double y1, 
                                                               double x2, double y2)
{
    uint r11, r12, r21, r22;
    uint g11, g12, g21, g22;
    uint b11, b12, b21, b22;
    std::tie(r11, g11, b11) = img(x1, y1);
    std::tie(r12, g12, b12) = img(x1, y2);
    std::tie(r21, g21, b21) = img(x2, y1);
    std::tie(r22, g22, b22) = img(x2, y2);

    uint r = r11 * (x2 - x) * (y2 - y) + r21 * (x - x1) * (y2 - y) +
             r12 * (x2 - x) * (y - y1) + r22 * (x - x1) * (y - y1);
    uint g = g11 * (x2 - x) * (y2 - y) + g21 * (x - x1) * (y2 - y) +
             g12 * (x2 - x) * (y - y1) + g22 * (x - x1) * (y - y1);
    uint b = b11 * (x2 - x) * (y2 - y) + b21 * (x - x1) * (y2 - y) +
             b12 * (x2 - x) * (y - y1) + b22 * (x - x1) * (y - y1);
    return std::make_tuple(r, g, b);
}

Image resize(const Image &srcImage, double scale) 
{
    Image result(std::floor(srcImage.n_rows * scale), std::floor(srcImage.n_cols * scale));
    double x, y, x1, x2, y1, y2;
    for (uint i = 0; i < result.n_rows; ++i) {
        x = i / scale;
        x1 = std::floor(x);
        for (uint j = 0; j < result.n_cols; ++j) {
            y = j / scale;
            y1 = std::floor(y);
            x2 = x1 + 1;
            y2 = y1 + 1;
            if (x2 >= srcImage.n_rows) {
                x2 = x1;
            }
            if (y2 >= srcImage.n_cols) {
                y2 = y1;
            }
            result(i, j) = bilinear_interp(srcImage, x, y, x1, y1, x2, y2);
        }
    }
    return result;
}

Matrix<uint> resize(const Matrix<uint> &srcImage, double scale) 
{
    Matrix<uint> result(std::floor(srcImage.n_rows * scale), std::floor(srcImage.n_cols * scale));
    double x, y, x1, x2, y1, y2;
    for (uint i = 0; i < result.n_rows; ++i) {
        x = i / scale;
        x1 = std::floor(x);
        for (uint j = 0; j < result.n_cols; ++j) {
            y = j / scale;
            y1 = std::floor(y);
            x2 = x1 + 1;
            y2 = y1 + 1;
            result(i, j) = bilinear_interp(srcImage, x, y, x1, y1, x2, y2);
        }
    }
    return result;
}

//returns best metric
double improve_shift (const Matrix<uint> &m1, const Matrix<uint> &m2, int &shift_r, int &shift_c)
{
    Matrix<uint> window1 = m1.submatrix(pyramid_radius, pyramid_radius, 
                                        m1.n_rows - 2 * pyramid_radius, m1.n_cols - 2 * pyramid_radius);
    double metric_min = DBL_MAX;
    for (int row = -2; row <= 2; ++row) {
        for (int col = -2; col <= 2; ++col) {

            if ((pyramid_radius + shift_r + row >= 0) && (pyramid_radius + shift_c + col >= 0)) {
                if ((pyramid_radius + shift_r + row < 2 * pyramid_radius) &&
                                             (pyramid_radius + shift_c + col < 2 * pyramid_radius)) {
                    Matrix<uint> window2 = m2.submatrix(pyramid_radius + shift_r + row,
                                                        pyramid_radius + shift_c + col, 
                                                        m2.n_rows - 2 * pyramid_radius, 
                                                        m2.n_cols - 2 * pyramid_radius);
                    double t = mse(window1, window2);
                    if (metric_min - t > EPS) {
                        metric_min = t;
                        shift_r += row;
                        shift_c += col;
                    }
                }
            }
        }
    }
    return metric_min;
}

double pyramid (const Matrix<uint> &img1, const Matrix<uint> &img2, int &shift_r, int &shift_c)
{

    if (std::min(img1.n_rows, img1.n_cols) / 2 < MINLEN) {

        return find_shift_mse(img1, img2, shift_r, shift_c);
    } else {
        pyramid(resize(img1, 0.5), resize(img2, 0.5), shift_r, shift_c);
        shift_r *= 2;
        shift_c *= 2;
        pyramid_radius *= 2;
        return improve_shift(img1, img2, shift_r, shift_c);
    }
}

Image sobel_x(Image srcImage) {
    Matrix<double> kernel = {{-1, 0, 1},
                             {-2, 0, 2},
                             {-1, 0, 1}};
    return custom(srcImage, kernel);
}

Image sobel_y(Image srcImage) {
    Matrix<double> kernel = {{ 1,  2,  1},
                             { 0,  0,  0},
                             {-1, -2, -1}};
    return custom(srcImage, kernel);
}


Image mirror_edges(const Image &img, uint rad)
{
    Image ans(img.n_rows + 2 * rad, img.n_cols + 2 * rad);

    for (uint i = 0; i < img.n_rows; ++i) {
        for (uint j = 0; j < img.n_cols; ++j) {
            ans(i + rad, j + rad) = img(i, j);
        }
    }
    for (uint i = 0; i < rad; ++i) {
        for (uint j = rad; j < ans.n_cols - rad; ++j) {
            ans(i, j) = img(rad - i - 1, j - rad);
            ans(ans.n_rows - i - 1, j) = img(img.n_rows + i - rad, j - rad);
        }
    }
    for(uint i = rad; i < ans.n_rows - rad; ++i) {
        for (uint j = 0; j < rad; ++j) {
            ans(i, j) = img(i - rad, rad - j - 1);
            ans(i, ans.n_cols - j - 1) = img(i - rad, img.n_cols + j - rad);
        }
    }
    for (uint i = 0; i < rad; ++i) {
        for (uint j = 0; j < rad; ++j) {
            ans(i, j) = ans(i, 2 * rad - j - 1);
            ans(i, ans.n_cols - j - 1) = ans(i, ans.n_cols - 2 * rad + j);
            ans(ans.n_rows - i - 1, j) = 
                        ans(ans.n_rows - i - 1, 2 * rad - j - 1);
            ans(ans.n_rows - i - 1, ans.n_cols - j - 1) = 
                        ans(ans.n_rows - i - 1, ans.n_cols - 2 * rad + j);
        }
    }
    return ans;
}

class UnsharpOp
{
public:
    static const int radius = 1;
    std::tuple<uint, uint, uint> operator () (const Image &src) const
    {
        const uint size = 2 * radius + 1;
        const Matrix<double> kernel = {{-1.0 / 6.0, -2.0 / 3.0, -1.0 / 6.0},
                                       {-2.0 / 3.0, 13.0 / 3.0, -2.0 / 3.0},
                                       {-1.0 / 6.0, -2.0 / 3.0, -1.0 / 6.0}};
        uint r, g, b;
        double sum_r = 0.0, sum_g = 0.0, sum_b = 0.0;
        for (uint i = 0; i < size; ++i) {
            for (uint j = 0; j < size; ++j) {
                std::tie(r, g, b) = src(i, j);
                sum_r += r * kernel(i, j);
                sum_g += g * kernel(i, j);
                sum_b += b * kernel(i, j);
            }
        }

        sum_r = sum_r < 255 ? sum_r : 255;
        sum_g = sum_g < 255 ? sum_g : 255;
        sum_b = sum_b < 255 ? sum_b : 255;
        sum_r = sum_r > 0 ? sum_r : 0;
        sum_g = sum_g > 0 ? sum_g : 0;
        sum_b = sum_b > 0 ? sum_b : 0;

        return std::make_tuple(sum_r, sum_g, sum_b);       
    }
};

Image unsharp(const Image &srcImage) {
    Image img = srcImage.unary_map(UnsharpOp());
    return img;
}

Image gray_world(const Image &srcImage)
{
    uint r, g, b, sum;
    uint sum_r = 0, sum_g = 0, sum_b = 0;
    for (uint i = 0; i < srcImage.n_rows; ++i) {
        for (uint j =  0; j < srcImage.n_cols; ++j) {
            std::tie(r, g, b) = srcImage(i, j);
            sum_r += r;
            sum_g += g;
            sum_b += b;
        }
    }
    uint size = srcImage.n_rows * srcImage.n_cols;
    sum_r /= size;
    sum_g /= size;
    sum_b /= size;

    sum = (sum_r + sum_g + sum_b);
    Image img(srcImage.n_rows, srcImage.n_cols);
    for (uint i = 0; i < srcImage.n_rows; ++i) {
        for (uint j =  0; j < srcImage.n_cols; ++j) {
            std::tie(r, g, b) = srcImage(i, j);
            r = r * sum / (3 * sum_r);
            g = g * sum / (3 * sum_g);
            b = b * sum / (3 * sum_b);

            r = r < 255 ? r : 255;
            g = g < 255 ? g : 255;
            b = b < 255 ? b : 255;
            img(i, j) = std::make_tuple(r, g, b);
        }
    }
    return img;
}

Image custom(Image srcImage, Matrix<double> kernel) {
    // Function custom is useful for making concrete linear filtrations
    // like gaussian or sobel. So, we assume that you implement custom
    // and then implement other filtrations using this function.
    // sobel_x and sobel_y are given as an example.
    return srcImage;
}

class Histogram
{
public:
    Histogram():
        fraction(0.01),
        total(0),
        maximum(0),
        minimum(255)
    {
        for (int i = 0; i < 256; ++i){
            histogram[i] = 0;
        }
    };
    uint calc_value (uint r, uint g, uint b)
    {
        const double coefR = 0.2125;
        const double coefG = 0.7154;
        const double coefB = 0.0721;
        double v = coefR * r + coefG * g + coefB * b;
        return std::round(v);
    }
    void get_histogram (const Image &img)
    {
        for (uint i = 0; i < img.n_rows; ++i){
            for (uint j = 0; j < img.n_cols; ++j){
                uint r, g, b;
                std::tie(r, g, b) = img(i, j);
                uint y = calc_value(r, g, b);
                ++histogram[y];
                ++total;
                if (y < minimum) {
                    minimum = y;
                }
                if (y > maximum) {
                    maximum = y;
                }
            }
        }
    }
    void cut_edges () 
    {
        uint delta = fraction * (maximum - minimum);
        maximum -= delta;
        minimum += delta;
    }
    void set_fraction (double v) {
        fraction = v;
    }
    uint get_min() {
        return minimum;
    }
    uint get_max() {
        return maximum;
    }
private:
    double fraction;
    int total;
    uint maximum;
    uint minimum;
    uint histogram[256];
};


Image autocontrast(const Image &srcImage, double fraction) { 
    Histogram h;
    h.get_histogram(srcImage);
    h.set_fraction(fraction);
    h.cut_edges();
    uint r, g, b;
    Image img(srcImage.n_rows, srcImage.n_cols);
    for (uint i = 0; i < srcImage.n_rows; ++i) {
        for (uint j = 0; j < srcImage.n_cols; ++j) {
            std::tie(r, g, b) = srcImage(i, j);
            uint value = h.calc_value(r, g, b);
            if (value < h.get_min()) {
               r = 0;
               g = 0;
               b = 0;
            } else {
                uint coef = (value - h.get_min()) * 255;
                r = r * coef / ((h.get_max() - h.get_min()) * value);
                g = g * coef / ((h.get_max() - h.get_min()) * value);
                b = b * coef / ((h.get_max() - h.get_min()) * value);
            }
            r = r < 256 ? r : 255;
            g = g < 256 ? g : 255;
            b = b < 256 ? b : 255;
            img(i, j) = std::make_tuple(r, g, b);
        }
    }
    return img;
}

Image gaussian(Image srcImage, double sigma, int radius)  {
    return srcImage;
}

Image gaussian_separable(Image srcImage, double sigma, int radius) {
    return srcImage;
}

uint find_median (std::vector<uint> &arr, uint half) {
    uint sum = 0;
    int i = 0; 
    while (sum < half) {
        sum += std::min(arr[i], half - sum);
        ++i;
    }
    --i;
    while (arr[i] == 0) {
        ++i;
    }
    return i;
}

void img_to_arrays (const Image &img, std::vector<uint> &r, std::vector<uint> &g, std::vector<uint> &b)
{
    uint red, green, blue;
    for (uint i = 0; i < img.n_rows; ++i) {
        for (uint j = 0; j < img.n_cols; ++j) {
            std::tie(red, green, blue) = img(i, j);
            ++r[red];
            ++g[green];
            ++b[blue];
        }
    }
}

std::tuple<uint, uint, uint> median_bucket_sort (const Image &img) {
    int size = 256;
    std::vector<uint> r(size, 0);
    std::vector<uint> g(size, 0);
    std::vector<uint> b(size, 0);
    uint half = std::round(img.n_rows * img.n_cols / 2.0);
    
    img_to_arrays(img, r, g, b);   
    
    uint red, green, blue;
    red = find_median(r, half);
    green = find_median(g, half);
    blue = find_median(b, half);
    return std::make_tuple(red, green, blue);
}

Image median(const Image &srcImage, uint radius) {
    Image result = mirror_edges(srcImage, radius);
    for (uint i = radius; i < srcImage.n_rows - radius; ++i) {
        for (uint j = radius; j < srcImage.n_cols - radius; ++j) {
            uint startRow = i - radius;
            uint startCol = j - radius;
            Image m = result.submatrix(startRow, startCol, radius * 2 + 1, radius * 2 + 1);
            result(i, j) = median_bucket_sort(m);
        }
    }
    return result.submatrix(radius, radius, result.n_rows - 2 * radius, result.n_cols - 2 * radius);
}

void initialize_window_hist (const Image &img, 
                             std::vector<uint> &r, std::vector<uint> &g, 
                             std::vector<uint> &b, uint rad)
{
    for (uint i = 0; i < 2 * rad + 1; ++i) {
        for (uint j = 0; j < 2 * rad + 1; ++j) {
            uint red, green, blue;
            std::tie(red, green, blue) = img(i, j);
            ++r[red];
            ++g[green];
            ++b[blue];
        }
    }
}

Image median_linear(const Image &src, uint radius) {
    Image srcImage = mirror_edges(src, radius);
    Image result(srcImage.n_rows - 2 * radius, srcImage.n_cols - 2 * radius);
    std::vector<uint> r(256, 0); // histograms of each channel
    std::vector<uint> g(256, 0);
    std::vector<uint> b(256, 0);
    initialize_window_hist(srcImage, r, g, b, radius);
    uint red, green, blue;
    uint half = std::round((2 * radius + 1) * (2 * radius + 1) / 2.0);
    for (uint i = radius; i < srcImage.n_rows - radius; ++i) {
        for (uint j = radius; j < srcImage.n_cols - radius; ++j) {
            for (uint k = 0; k < 2 * radius + 1; ++k) {
                if (j == radius) {
                    break;
                }
                std::tie(red, green, blue) = srcImage(i + k - radius, j - radius - 1);
                --r[red];
                --g[green];
                --b[blue];
                std::tie(red, green, blue) = srcImage(i + k - radius, j + radius);
                ++r[red];
                ++g[green];
                ++b[blue];
            }
            red = find_median(r, half);
            green = find_median(g, half);
            blue = find_median(b, half);
            result(i - radius, j - radius) = std::make_tuple(red, green, blue);
        }
        ++i;
        if (i == srcImage.n_rows - radius) {
            break;
        }
        for (uint k = 0; k < 2 * radius + 1; ++k) {
            std::tie(red, green, blue) = srcImage(i - radius - 1, srcImage.n_cols + k - 2 * radius - 1);
            --r[red];
            --g[green];
            --b[blue];
            std::tie(red, green, blue) = srcImage(i + radius, srcImage.n_cols + k - 2 * radius - 1);
            ++r[red];
            ++g[green];
            ++b[blue];
        }
        red = find_median(r, half);
        green = find_median(g, half);
        blue = find_median(b, half);
        result(i - radius, result.n_cols - 1) = std::make_tuple(red, green, blue);

        for (uint j = srcImage.n_cols - radius - 1; j >= radius; --j) {
            for (uint k = 0; k < 2 * radius + 1; ++k) {
                if (j == srcImage.n_cols - radius - 1) {
                    break;
                }
                std::tie(red, green, blue) = srcImage(i + k - radius, j + radius + 1);
                --r[red];
                --g[green];
                --b[blue];
                std::tie(red, green, blue) = srcImage(i + k - radius, j - radius);
                ++r[red];
                ++g[green];
                ++b[blue];
            }
            red = find_median(r, half);
            green = find_median(g, half);
            blue = find_median(b, half);
            result(i - radius, j - radius) = std::make_tuple(red, green, blue);
        } 
        if (i == srcImage.n_rows - radius - 1) {
            break;
        }
        for (uint k = 0; k < 2 * radius + 1; ++k) {
            std::tie(red, green, blue) = srcImage(i - radius, 2 * radius - k);
            --r[red];
            --g[green];
            --b[blue];
            std::tie(red, green, blue) = srcImage(i + radius + 1, 2 * radius - k);
            ++r[red];
            ++g[green];
            ++b[blue];
        }

    }
    return result;
}

Image median_const(Image srcImage, int radius) {
    return srcImage;
}

Image canny(Image srcImage, int threshold1, int threshold2) {
    return srcImage;
}

Image align(const Image &srcImage, bool isPostprocessing, std::string postprocessingType, double fraction, bool isMirror, 
            bool isInterp, bool isSubpixel, double subScale)
{
    Image tmp;
    if (isSubpixel) {
        tmp = resize(srcImage, subScale);
    } else {
        tmp = srcImage;
    }

    uint destRows = tmp.n_rows / 3;
    uint destCols = tmp.n_cols;
    Matrix<uint> greyImg = get_greyscale(tmp);

    Matrix<uint> redImg(destRows, destCols), blueImg(destRows, destCols), greenImg(destRows, destCols);
    split3(greyImg, redImg, greenImg, blueImg);
    
    
    double metricRB, metricRG, metricBG;
    int shiftRowRB, shiftColRB, shiftRowRG, shiftColRG, shiftRowBG, shiftColBG;
    if (std::min(destRows, destCols) / 2 > MINLEN) {
        pyramid_radius = RAD;
        metricRB = pyramid(redImg, blueImg, shiftRowRB, shiftColRB);
        pyramid_radius = RAD;
        metricRG = pyramid(redImg, greenImg, shiftRowRG, shiftColRG);
        pyramid_radius = RAD;
        metricBG = pyramid(blueImg, greenImg, shiftRowBG, shiftColBG);
    } else {
        metricRB = find_shift_mse(redImg, blueImg, shiftRowRB, shiftColRB);
        metricRG = find_shift_mse(redImg, greenImg, shiftRowRG, shiftColRG);
        metricBG = find_shift_mse(blueImg, greenImg, shiftRowBG, shiftColBG);

//      metricRB = find_shift_cc(redImg, blueImg, shiftRowRB, shiftColRB);
  //    metricRG = find_shift_cc(redImg, greenImg, shiftRowRG, shiftColRG);
    //  metricBG = find_shift_cc(blueImg, greenImg, shiftRowBG, shiftColBG);
    }

    if ((metricRB - metricRG > EPS) && (metricRB - metricBG > EPS)){ // RB - max
        cut_before_align(greenImg, redImg, blueImg, shiftRowRG, shiftColRG, shiftRowBG, shiftColBG);
    } else if ((metricRG - metricRB > EPS) && (metricRG - metricBG > EPS)){ // RG - max
        cut_before_align(blueImg, redImg, greenImg, shiftRowRB, shiftColRB, -shiftRowBG, -shiftColBG);
    } else if ((metricBG - metricRB > EPS) && (metricBG - metricRG > EPS)){ // BG - max
        cut_before_align(redImg, blueImg, greenImg, -shiftRowRB, -shiftColRB, -shiftRowRG, -shiftColRG);
    } 
 

    destRows = redImg.n_rows;
    destCols = redImg.n_cols;

    Image ansImage = Image(destRows, destCols);

    for (uint i = 0; i < destRows; ++i) {
        for (uint j = 0; j < destCols; ++j) {
            ansImage(i, j) = std::make_tuple(redImg(i, j), greenImg(i, j), blueImg(i, j));
        }
    }
    if (isSubpixel) {
        ansImage = resize(ansImage, 1 / subScale);
    }

    if (isPostprocessing) {
         if (postprocessingType == "--gray-world") {
            ansImage = gray_world(ansImage);
         } else if(postprocessingType == "--unsharp") {
             if (isMirror) {
                 ansImage = mirror_edges(ansImage, 1);
             }
             ansImage = unsharp(ansImage);
             if (isMirror) {
                 ansImage = ansImage.submatrix(1, 1, ansImage.n_rows - 2, ansImage.n_cols - 2);
             }
         } else if (postprocessingType == "--autocontrast") {
            ansImage = autocontrast(ansImage, fraction);
         }
    }

    
    return ansImage;
}

