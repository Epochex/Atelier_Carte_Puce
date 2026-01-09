#include <opencv2/videoio.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cmath>
#include <chrono>
#include <iostream>
#include <limits>
#include <string>
#include <tuple>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Utils
static int clampi(int v, int lo, int hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

static int degBin(float radians) {
    float deg = radians * 180.0f / float(M_PI);
    int b = (int)std::lround(deg);
    b = b % 360;
    if (b < 0) b += 360;
    return b;
}

// image en gris
struct GrayImage {
    int w = 0, h = 0;
    std::vector<uint8_t> p;

    uint8_t& at(int y, int x) { return p[(size_t)y * (size_t)w + (size_t)x]; }
    uint8_t  at(int y, int x) const { return p[(size_t)y * (size_t)w + (size_t)x]; }
};

static GrayImage makeGray(int w, int h, uint8_t value) {
    GrayImage g;
    g.w = w;
    g.h = h;
    g.p.assign((size_t)w * (size_t)h, value);
    return g;
}

// BGR -> Gray manuel
static GrayImage bgrToGray(const cv::Mat& frame) {
    GrayImage g;
    g.w = frame.cols;
    g.h = frame.rows;
    g.p.resize((size_t)g.w * (size_t)g.h);

    for (int y = 0; y < g.h; ++y) {
        const uint8_t* row = frame.ptr<uint8_t>(y);
        for (int x = 0; x < g.w; ++x) {
            uint8_t b = row[3 * x + 0];
            uint8_t gg = row[3 * x + 1];
            uint8_t r = row[3 * x + 2];
            int gray = (299 * (int)r + 587 * (int)gg + 114 * (int)b) / 1000;
            g.at(y, x) = (uint8_t)clampi(gray, 0, 255);
        }
    }
    return g;
}

// Sobel maison
struct Gradients {
    int w = 0, h = 0;
    std::vector<int16_t> gx;
    std::vector<int16_t> gy;
    std::vector<uint16_t> mag; // |gx| + |gy|
    uint16_t Mag(int y, int x) const { return mag[(size_t)y * (size_t)w + (size_t)x]; }
};

static Gradients sobelManual(const GrayImage& g) {
    Gradients out;
    out.w = g.w;
    out.h = g.h;
    out.gx.assign((size_t)out.w * (size_t)out.h, 0);
    out.gy.assign((size_t)out.w * (size_t)out.h, 0);
    out.mag.assign((size_t)out.w * (size_t)out.h, 0);

    for (int y = 1; y < g.h - 1; ++y) {
        for (int x = 1; x < g.w - 1; ++x) {
            int a00 = g.at(y - 1, x - 1), a01 = g.at(y - 1, x), a02 = g.at(y - 1, x + 1);
            int a10 = g.at(y,     x - 1),                         a12 = g.at(y,     x + 1);
            int a20 = g.at(y + 1, x - 1), a21 = g.at(y + 1, x), a22 = g.at(y + 1, x + 1);

            int gx = (-a00 + a02) + (-2 * a10 + 2 * a12) + (-a20 + a22);
            int gy = (-a00 - 2 * a01 - a02) + (a20 + 2 * a21 + a22);

            out.gx[(size_t)y * (size_t)out.w + (size_t)x] = (int16_t)gx;
            out.gy[(size_t)y * (size_t)out.w + (size_t)x] = (int16_t)gy;

            int m = std::abs(gx) + std::abs(gy);
            out.mag[(size_t)y * (size_t)out.w + (size_t)x] = (uint16_t)clampi(m, 0, 65535);
        }
    }
    return out;
}

// Accumulateur + ROI
struct AccuImage {
    int w = 0, h = 0;
    std::vector<uint16_t> a;
    uint16_t& at(int y, int x) { return a[(size_t)y * (size_t)w + (size_t)x]; }
    uint16_t  at(int y, int x) const { return a[(size_t)y * (size_t)w + (size_t)x]; }
};

static AccuImage makeAccu(int w, int h) {
    AccuImage A;
    A.w = w;
    A.h = h;
    A.a.assign((size_t)w * (size_t)h, 0);
    return A;
}

struct Roi { int x0=0,y0=0,x1=0,y1=0; };

static Roi clampRoi(Roi r, int w, int h) {
    r.x0 = clampi(r.x0, 0, w);
    r.x1 = clampi(r.x1, 0, w);
    r.y0 = clampi(r.y0, 0, h);
    r.y1 = clampi(r.y1, 0, h);

    if (r.x1 < r.x0) std::swap(r.x0, r.x1);
    if (r.y1 < r.y0) std::swap(r.y0, r.y1);
    return r;
}

static bool roiValid(const Roi& r) {
    int ww = r.x1 - r.x0;
    int hh = r.y1 - r.y0;
    if (ww >= 8 && hh >= 8) return true;
    return false;
}

// R-table (alpha -> offsets dx,dy)
struct Offset { int dx; int dy; };
using LUTOffsets = std::array<std::vector<Offset>, 360>;

static LUTOffsets buildLUTOffsetsFromTemplate(const GrayImage& templ, int edgeThresholdMag, int maxPerBin) {
    LUTOffsets lut;
    for (int i = 0; i < 360; ++i) lut[i].clear();

    Gradients tg = sobelManual(templ);
    int cx = templ.w / 2;
    int cy = templ.h / 2;

    for (int y = 1; y < templ.h - 1; ++y) {
        for (int x = 1; x < templ.w - 1; ++x) {
            if (tg.Mag(y, x) < edgeThresholdMag) continue;

            float gx = (float)tg.gx[(size_t)y * (size_t)templ.w + (size_t)x];
            float gy = (float)tg.gy[(size_t)y * (size_t)templ.w + (size_t)x];
            if (gx == 0.0f && gy == 0.0f) continue;

            int aBin = degBin(std::atan2(gy, gx));

            Offset off;
            off.dx = cx - x;
            off.dy = cy - y;

            if ((int)lut[aBin].size() < maxPerBin) lut[aBin].push_back(off);
        }
    }
    return lut;
}

static AccuImage voteOffsets(const Gradients& grads, const LUTOffsets& lut, int edgeThresholdMag, Roi roiIn) {
    AccuImage accu = makeAccu(grads.w, grads.h);

    Roi roi = clampRoi(roiIn, accu.w, accu.h);
    if (!roiValid(roi)) return accu;

    int x0 = std::max(1, roi.x0);
    int x1 = std::min(accu.w - 2, roi.x1);
    int y0 = std::max(1, roi.y0);
    int y1 = std::min(accu.h - 2, roi.y1);

    for (int y = y0; y < y1; ++y) {
        for (int x = x0; x < x1; ++x) {
            if (grads.Mag(y, x) < edgeThresholdMag) continue;

            float gx = (float)grads.gx[(size_t)y * (size_t)accu.w + (size_t)x];
            float gy = (float)grads.gy[(size_t)y * (size_t)accu.w + (size_t)x];
            if (gx == 0.0f && gy == 0.0f) continue;

            int aBin = degBin(std::atan2(gy, gx));
            const std::vector<Offset>& offs = lut[aBin];
            if (offs.empty()) continue;

            for (size_t i = 0; i < offs.size(); ++i) {
                int cx = x + offs[i].dx;
                int cy = y + offs[i].dy;
                if (cx >= 0 && cx < accu.w && cy >= 0 && cy < accu.h) {
                    uint16_t v = accu.at(cy, cx);
                    if (v < std::numeric_limits<uint16_t>::max()) accu.at(cy, cx) = (uint16_t)(v + 1);
                }
            }
        }
    }
    return accu;
}

// Pic + barycentre local
struct PeakBary {
    bool ok = false;
    int px = -1, py = -1;
    uint16_t peak = 0;
    float bx = 0.0f, by = 0.0f;
};

static PeakBary localBarycenterAroundMax(const AccuImage& accu, int radius) {
    PeakBary out;

    int px = -1, py = -1;
    uint16_t best = 0;
    for (int y = 0; y < accu.h; ++y) {
        for (int x = 0; x < accu.w; ++x) {
            uint16_t v = accu.at(y, x);
            if (v > best) { best = v; px = x; py = y; }
        }
    }
    if (px < 0) return out;
    if (best == 0) return out;

    double sw = 0.0, sx = 0.0, sy = 0.0;
    int x0 = std::max(0, px - radius);
    int x1 = std::min(accu.w - 1, px + radius);
    int y0 = std::max(0, py - radius);
    int y1 = std::min(accu.h - 1, py + radius);

    for (int y = y0; y <= y1; ++y) {
        for (int x = x0; x <= x1; ++x) {
            uint16_t w = accu.at(y, x);
            sw += (double)w;
            sx += (double)x * (double)w;
            sy += (double)y * (double)w;
        }
    }
    if (sw <= 0.0) return out;

    out.ok = true;
    out.px = px;
    out.py = py;
    out.peak = best;
    out.bx = (float)(sx / sw);
    out.by = (float)(sy / sw);
    return out;
}

// TopK pics + NMS (pour 2 yeux)
struct PeakPoint {
    int x = -1;
    int y = -1;
    uint16_t v = 0;
    float bx = 0.0f;
    float by = 0.0f;
};

static void suppressDisk(AccuImage& A, int cx, int cy, int r) {
    int rr = r * r;
    int x0 = std::max(0, cx - r);
    int x1 = std::min(A.w - 1, cx + r);
    int y0 = std::max(0, cy - r);
    int y1 = std::min(A.h - 1, cy + r);

    for (int y = y0; y <= y1; ++y) {
        for (int x = x0; x <= x1; ++x) {
            int dx = x - cx;
            int dy = y - cy;
            if (dx*dx + dy*dy <= rr) A.at(y, x) = 0;
        }
    }
}

static std::vector<PeakPoint> topKPeaksWithBary(AccuImage A, int k, int nmsRadius, int baryRadius, uint16_t minVal) {
    std::vector<PeakPoint> out;

    for (int i = 0; i < k; ++i) {
        PeakBary b = localBarycenterAroundMax(A, baryRadius);
        if (!b.ok) break;
        if (b.peak < minVal) break;

        PeakPoint p;
        p.x = b.px; p.y = b.py; p.v = b.peak;
        p.bx = b.bx; p.by = b.by;
        out.push_back(p);

        suppressDisk(A, b.px, b.py, nmsRadius);
    }
    return out;
}

static bool pickBestEyePair(
    const std::vector<PeakPoint>& peaks,
    int faceX, int faceY,
    int minDx, int maxDx, int maxDy,
    PeakPoint& leftEye, PeakPoint& rightEye
) {
    bool found = false;
    uint32_t bestSum = 0;

    for (size_t i = 0; i < peaks.size(); ++i) {
        for (size_t j = i + 1; j < peaks.size(); ++j) {
            int ax = (int)std::lround(peaks[i].bx);
            int ay = (int)std::lround(peaks[i].by);
            int bx = (int)std::lround(peaks[j].bx);
            int by = (int)std::lround(peaks[j].by);

            int dx = std::abs(ax - bx);
            int dy = std::abs(ay - by);

            if (dx < minDx) continue;
            if (dx > maxDx) continue;
            if (dy > maxDy) continue;

            // yeux au-dessus du centre visage
            if (ay >= faceY) continue;
            if (by >= faceY) continue;

            uint32_t sum = (uint32_t)peaks[i].v + (uint32_t)peaks[j].v;
            if (!found || sum > bestSum) {
                found = true;
                bestSum = sum;

                if (ax <= bx) { leftEye = peaks[i]; rightEye = peaks[j]; }
                else { leftEye = peaks[j]; rightEye = peaks[i]; }
            }
        }
    }
    return found;
}

// Templates artificiels (contours noirs sur fond blanc)
static GrayImage makeEllipseTemplate(int w, int h, float rx, float ry) {
    GrayImage t = makeGray(w, h, 255);
    float cx = (float)(w / 2);
    float cy = (float)(h / 2);

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            float dx = (float)x - cx;
            float dy = (float)y - cy;
            float v = (dx*dx)/(rx*rx) + (dy*dy)/(ry*ry);
            if (std::fabs(v - 1.0f) < 0.03f) t.at(y, x) = 0;
        }
    }
    return t;
}

static GrayImage makeCircleTemplate(int w, int h, float r) {
    GrayImage t = makeGray(w, h, 255);
    float cx = (float)(w / 2);
    float cy = (float)(h / 2);

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            float dx = (float)x - cx;
            float dy = (float)y - cy;
            float d = std::sqrt(dx*dx + dy*dy);
            if (std::fabs(d - r) < 3.0f) t.at(y, x) = 0;
        }
    }
    return t;
}

// Images artificielles pour --test
static GrayImage makeArtificialEllipseImage(int w, int h) {
    GrayImage img = makeGray(w, h, 255);
    int cx = w / 2;
    int cy = h / 2;

    float ry = 0.35f * (float)h;
    float rx = 0.5f * ry;

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            float dx = (float)(x - cx);
            float dy = (float)(y - cy);
            float v = (dx*dx)/(rx*rx) + (dy*dy)/(ry*ry);
            if (std::fabs(v - 1.0f) < 0.03f) img.at(y, x) = 0;
        }
    }
    return img;
}

static GrayImage makeArtificialCircleImage(int w, int h) {
    GrayImage img = makeGray(w, h, 255);
    int cx = w / 2;
    int cy = h / 2;
    float r = 0.30f * (float)std::min(w, h);

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            float dx = (float)(x - cx);
            float dy = (float)(y - cy);
            float d = std::sqrt(dx*dx + dy*dy);
            if (std::fabs(d - r) < 3.0f) img.at(y, x) = 0;
        }
    }
    return img;
}

// Modèles multi-échelle
struct FaceModel {
    int rx = 0;
    int ry = 0;
    LUTOffsets lut;
};

struct EyeModel {
    int r = 0;
    LUTOffsets lut;
};

// Détection visage + 2 yeux
struct FaceEyes {
    bool faceOk = false;
    int faceX = -1, faceY = -1;
    uint16_t faceScore = 0;
    int faceRx = 0;
    int faceRy = 0;

    bool eyesOk = false;
    int ex1=-1, ey1=-1;
    int ex2=-1, ey2=-1;
    uint16_t eyeScoreSum = 0;
    int eyeR = 0;
};

static FaceEyes detectFaceEyes(
    const GrayImage& gray,
    const std::vector<FaceModel>& faceModels,
    const std::vector<EyeModel>& eyeModels,
    int EDGE_FACE,
    int EDGE_EYE,
    uint16_t FACE_MIN_SCORE,
    uint16_t EYE_MIN_PEAK
) {
    FaceEyes out;
    Gradients grads = sobelManual(gray);

    Roi faceRoi;
    faceRoi.x0 = (int)(gray.w * 0.10);
    faceRoi.x1 = (int)(gray.w * 0.90);
    faceRoi.y0 = (int)(gray.h * 0.05);
    faceRoi.y1 = (int)(gray.h * 0.95);

    bool bestOk = false;
    PeakBary bestPeak;
    int bestRx = 0, bestRy = 0;

    for (size_t i = 0; i < faceModels.size(); ++i) {
        AccuImage A = voteOffsets(grads, faceModels[i].lut, EDGE_FACE, faceRoi);
        PeakBary p = localBarycenterAroundMax(A, 22);
        if (!p.ok) continue;
        if (p.peak < FACE_MIN_SCORE) continue;

        if (!bestOk || p.peak > bestPeak.peak) {
            bestOk = true;
            bestPeak = p;
            bestRx = faceModels[i].rx;
            bestRy = faceModels[i].ry;
        }
    }

    if (!bestOk) return out;

    out.faceOk = true;
    out.faceX = (int)std::lround(bestPeak.bx);
    out.faceY = (int)std::lround(bestPeak.by);
    out.faceScore = bestPeak.peak;
    out.faceRx = bestRx;
    out.faceRy = bestRy;

    Roi eyeRoi;
    eyeRoi.x0 = out.faceX - (int)std::lround((double)out.faceRx * 2.2);
    eyeRoi.x1 = out.faceX + (int)std::lround((double)out.faceRx * 2.2);
    eyeRoi.y0 = out.faceY - (int)std::lround((double)out.faceRy * 0.95);
    eyeRoi.y1 = out.faceY - (int)std::lround((double)out.faceRy * 0.30);
    eyeRoi = clampRoi(eyeRoi, gray.w, gray.h);
    if (!roiValid(eyeRoi)) return out;

    bool pairOk = false;
    uint32_t bestSum = 0;
    PeakPoint bestL, bestR;
    int bestRval = 0;

    for (size_t i = 0; i < eyeModels.size(); ++i) {
        AccuImage A = voteOffsets(grads, eyeModels[i].lut, EDGE_EYE, eyeRoi);

        std::vector<PeakPoint> peaks = topKPeaksWithBary(A, 6, 14, 10, EYE_MIN_PEAK);

        PeakPoint Lp, Rp;

        int minDx = std::max(25, (int)std::lround((double)out.faceRx * 0.8));
        int maxDx = std::min(gray.w, (int)std::lround((double)out.faceRx * 2.8));
        int maxDy = std::max(12, (int)std::lround((double)out.faceRy * 0.18));

        if (pickBestEyePair(peaks, out.faceX, out.faceY, minDx, maxDx, maxDy, Lp, Rp)) {
            uint32_t sum = (uint32_t)Lp.v + (uint32_t)Rp.v;
            if (!pairOk || sum > bestSum) {
                pairOk = true;
                bestSum = sum;
                bestL = Lp;
                bestR = Rp;
                bestRval = eyeModels[i].r;
            }
        }
    }

    if (!pairOk) return out;

    out.eyesOk = true;
    out.ex1 = (int)std::lround(bestL.bx);
    out.ey1 = (int)std::lround(bestL.by);
    out.ex2 = (int)std::lround(bestR.bx);
    out.ey2 = (int)std::lround(bestR.by);
    out.eyeScoreSum = (uint16_t)clampi((int)bestSum, 0, 65535);
    out.eyeR = bestRval;

    return out;
}

// MAIN
int main(int argc, char** argv) {
    bool doTest = false;
    bool doImage = false;
    std::string imagePath;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--test") doTest = true;
        else if (a == "--image") {
            if (i + 1 < argc) { doImage = true; imagePath = argv[i + 1]; i++; }
        }
    }

    const int EDGE_FACE = 140;
    const int EDGE_EYE  = 75;
    const uint16_t FACE_MIN_SCORE = 14;
    const uint16_t EYE_MIN_PEAK   = 5;

    // Multi-échelle visage (ellipse)
    std::vector<FaceModel> faceModels;
    {
        const int scales[][2] = {
            {25, 45}, {30, 55}, {35, 65}, {45, 85}, {55, 105}, {65, 125}, {75, 145}
        };
        for (int i = 0; i < 7; ++i) {
            int rx = scales[i][0];
            int ry = scales[i][1];

            int tw = 2 * rx + 60;
            int th = 2 * ry + 60;

            GrayImage t = makeEllipseTemplate(tw, th, (float)rx, (float)ry);
            LUTOffsets lut = buildLUTOffsetsFromTemplate(t, 50, 220);

            FaceModel fm;
            fm.rx = rx;
            fm.ry = ry;
            fm.lut = lut;
            faceModels.push_back(fm);
        }
    }

    // Multi-échelle yeux (cercle)
    std::vector<EyeModel> eyeModels;
    {
        for (int r = 6; r <= 18; r += 2) {
            int tw = 2 * r + 40;
            int th = 2 * r + 40;

            GrayImage t = makeCircleTemplate(tw, th, (float)r);
            LUTOffsets lut = buildLUTOffsetsFromTemplate(t, 40, 220);

            EyeModel em;
            em.r = r;
            em.lut = lut;
            eyeModels.push_back(em);
        }
    }

    // --test
    if (doTest) {
        std::cout << "[TEST] images artificielles + LUT + detection\n";

        GrayImage imgE = makeArtificialEllipseImage(640, 480);
        FaceEyes rE = detectFaceEyes(imgE, faceModels, eyeModels, EDGE_FACE, EDGE_EYE, FACE_MIN_SCORE, EYE_MIN_PEAK);
        std::cout << "Expected ellipse center=(" << imgE.w/2 << "," << imgE.h/2 << ")\n";
        if (rE.faceOk) {
            std::cout << "Detected ellipse center=(" << rE.faceX << "," << rE.faceY
                      << ") score=" << rE.faceScore
                      << " scale=(" << rE.faceRx << "," << rE.faceRy << ")\n";
        } else {
            std::cout << "Detected ellipse: NOTFOUND\n";
        }

        GrayImage imgC = makeArtificialCircleImage(320, 320);
        Gradients gC = sobelManual(imgC);
        Roi fullC; fullC.x0=0; fullC.y0=0; fullC.x1=imgC.w; fullC.y1=imgC.h;

        bool ok = false;
        int bestX = -1, bestY = -1;
        uint16_t bestSc = 0;
        int bestR = 0;

        for (size_t i = 0; i < eyeModels.size(); ++i) {
            AccuImage A = voteOffsets(gC, eyeModels[i].lut, EDGE_EYE, fullC);
            PeakBary p = localBarycenterAroundMax(A, 17);
            if (!p.ok) continue;
            if (!ok || p.peak > bestSc) {
                ok = true;
                bestSc = p.peak;
                bestX = (int)std::lround(p.bx);
                bestY = (int)std::lround(p.by);
                bestR = eyeModels[i].r;
            }
        }

        std::cout << "Expected circle center=(" << imgC.w/2 << "," << imgC.h/2 << ")\n";
        if (ok) {
            std::cout << "Detected circle center=(" << bestX << "," << bestY
                      << ") score=" << bestSc << " r=" << bestR << "\n";
        } else {
            std::cout << "Detected circle: NOTFOUND\n";
        }

        std::cout << "[TEST] fini.\n";
        return 0;
    }

    // --image
    if (doImage) {
        cv::Mat img = cv::imread(imagePath);
        if (img.empty()) {
            std::cerr << "Erreur: impossible de lire l'image: " << imagePath << "\n";
            return 1;
        }
        if (img.channels() != 3) {
            std::cerr << "Erreur: image doit etre en BGR (3 canaux)\n";
            return 1;
        }

        GrayImage gray = bgrToGray(img);
        FaceEyes r = detectFaceEyes(gray, faceModels, eyeModels, EDGE_FACE, EDGE_EYE, FACE_MIN_SCORE, EYE_MIN_PEAK);

        if (r.faceOk) {
            std::cout << "OK !\n";
            std::cout << "Face=(" << r.faceX << "," << r.faceY << ") score=" << r.faceScore
                      << " scale=(" << r.faceRx << "," << r.faceRy << ")\n";
        } else {
            std::cout << "Face=NOTFOUND\n";
        }
        if (r.eyesOk) {
            std::cout << "Eyes=(" << r.ex1 << "," << r.ey1 << ") (" << r.ex2 << "," << r.ey2 << ") "
                      << "eyeScoreSum=" << r.eyeScoreSum << " r=" << r.eyeR << "\n";
        } else {
            std::cout << "Eyes=NOTFOUND\n";
        }
        return 0;
    }

    // Webcam
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Camera non ouverte.\n";
        return 1;
    }

    cv::Mat frame;
    cap >> frame;
    if (frame.empty()) {
        std::cerr << "Impossible de lire une frame.\n";
        return 1;
    }

    std::cout << "Resolution camera: " << frame.cols << "x" << frame.rows << "\n";
    std::cout << "Detection en cours. Ctrl+C pour quitter.\n";

    // OK ! au max une fois toutes les 5 secondes
    using Clock = std::chrono::steady_clock;
    auto lastOkTime = Clock::now() - std::chrono::seconds(5);
    const int OK_COOLDOWN_SEC = 5;

    int frameCount = 0;
    while (true) {
        cap >> frame;
        if (frame.empty()) continue;
        if (frame.channels() != 3) continue;

        GrayImage gray = bgrToGray(frame);
        FaceEyes r = detectFaceEyes(gray, faceModels, eyeModels, EDGE_FACE, EDGE_EYE, FACE_MIN_SCORE, EYE_MIN_PEAK);

        // Affiche OK ! au maximum une fois par tranche de 5 secondes
        auto now = Clock::now();
        if (r.faceOk) {
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - lastOkTime).count();
            if (elapsed >= OK_COOLDOWN_SEC) {
                std::cout << "OK !\n";
                lastOkTime = now;
            }
        }

        if (frameCount % 30 == 0) {
            if (r.faceOk) {
                std::cout << "Face=(" << r.faceX << "," << r.faceY << ") score=" << r.faceScore
                          << " scale=(" << r.faceRx << "," << r.faceRy << ")";
                if (r.eyesOk) {
                    std::cout << " Eyes=(" << r.ex1 << "," << r.ey1 << ") (" << r.ex2 << "," << r.ey2 << ") "
                              << "eyeScoreSum=" << r.eyeScoreSum << " r=" << r.eyeR;
                } else {
                    std::cout << " Eyes=NOTFOUND";
                }
                std::cout << "\n";
            } else {
                std::cout << "Face=NOTFOUND\n";
            }
        }

        frameCount++;
    }

    return 0;
}
