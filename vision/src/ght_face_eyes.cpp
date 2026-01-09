#include <opencv2/videoio.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cmath>
#include <chrono>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// utils
static int clampInt(int v, int minV, int maxV) {
    if (v < minV) return minV;
    if (v > maxV) return maxV;
    return v;
}

static int binDeg(float radians) {
    float deg = radians * 180.0f / float(M_PI);
    int b = (int)std::lround(deg);
    b = b % 360;
    if (b < 0) b += 360;
    return b;
}

// image en gris
struct grayImage {
    int w = 0, h = 0;
    std::vector<uint8_t> p;

    uint8_t& at(int y, int x) { return p[(size_t)y * (size_t)w + (size_t)x]; }
    uint8_t  at(int y, int x) const { return p[(size_t)y * (size_t)w + (size_t)x]; }
};

static grayImage makeGris(int w, int h, uint8_t value) {
    grayImage g;
    g.w = w;
    g.h = h;
    g.p.assign((size_t)w * (size_t)h, value);
    return g;
}

// BGR en gris
static grayImage bgrToGray(const cv::Mat& frame) {
    grayImage g;
    g.w = frame.cols;
    g.h = frame.rows;
    g.p.assign((size_t)g.w * (size_t)g.h, 0);

    for (int y = 0; y < g.h; ++y) {
        const cv::Vec3b* row = frame.ptr<cv::Vec3b>(y);
        for (int x = 0; x < g.w; ++x) {
            // BGR
            const int b = row[x][0];
            const int gg = row[x][1];
            const int r = row[x][2];
            int v = (int)std::lround(0.114 * b + 0.587 * gg + 0.299 * r);
            g.at(y, x) = (uint8_t)clampInt(v, 0, 255);
        }
    }
    return g;
}

// Conversion to cv::Mat (debug)
static cv::Mat toMatGray8(const grayImage& g) {
    cv::Mat m(g.h, g.w, CV_8UC1);
    for (int y = 0; y < g.h; ++y) {
        uint8_t* row = m.ptr<uint8_t>(y);
        for (int x = 0; x < g.w; ++x) row[x] = g.at(y, x);
    }
    return m;
}

struct ChampGradient {
    int w = 0, h = 0;
    std::vector<int16_t> gradX;
    std::vector<int16_t> gradY;
    std::vector<uint16_t> mag;
    uint16_t Mag(int y, int x) const { return mag[(size_t)y * (size_t)w + (size_t)x]; }
};

static ChampGradient sobel(const grayImage& g) {
    ChampGradient out;
    out.w = g.w; out.h = g.h;
    out.gradX.assign((size_t)out.w * (size_t)out.h, 0);
    out.gradY.assign((size_t)out.w * (size_t)out.h, 0);
    out.mag.assign((size_t)out.w * (size_t)out.h, 0);

    for (int y = 1; y < g.h - 1; ++y) {
        for (int x = 1; x < g.w - 1; ++x) {
            int gx =
                -1 * g.at(y - 1, x - 1) + 1 * g.at(y - 1, x + 1) +
                -2 * g.at(y, x - 1)     + 2 * g.at(y, x + 1) +
                -1 * g.at(y + 1, x - 1) + 1 * g.at(y + 1, x + 1);

            int gy =
                -1 * g.at(y - 1, x - 1) + -2 * g.at(y - 1, x) + -1 * g.at(y - 1, x + 1) +
                 1 * g.at(y + 1, x - 1) +  2 * g.at(y + 1, x) +  1 * g.at(y + 1, x + 1);

            out.gradX[(size_t)y * (size_t)out.w + (size_t)x] = (int16_t)gx;
            out.gradY[(size_t)y * (size_t)out.w + (size_t)x] = (int16_t)gy;

            int m = (int)std::lround(std::sqrt((double)gx * (double)gx + (double)gy * (double)gy));
            out.mag[(size_t)y * (size_t)out.w + (size_t)x] = (uint16_t)clampInt(m, 0, 65535);
        }
    }
    return out;
}

static cv::Mat toMatMag8(const ChampGradient& cg) {
    uint16_t maxV = 1;
    for (auto v : cg.mag) maxV = std::max<uint16_t>(maxV, v);

    cv::Mat m(cg.h, cg.w, CV_8UC1);
    for (int y = 0; y < cg.h; ++y) {
        uint8_t* row = m.ptr<uint8_t>(y);
        for (int x = 0; x < cg.w; ++x) {
            uint16_t v = cg.mag[(size_t)y * (size_t)cg.w + (size_t)x];
            row[x] = (uint8_t)((uint32_t)v * 255u / (uint32_t)maxV);
        }
    }
    return m;
}

// R-Table (GHT)
struct Decalage { int dx = 0, dy = 0; };
using RTable = std::array<std::vector<Decalage>, 360>;

static grayImage templateEllipse(int w, int h, float rx, float ry) {
    grayImage t = makeGris(w, h, 0);
    int cx = w / 2;
    int cy = h / 2;

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            float dx = (float)(x - cx);
            float dy = (float)(y - cy);
            float v = (dx*dx) / (rx*rx) + (dy*dy) / (ry*ry);
            if (std::abs(v - 1.0f) < 0.05f) t.at(y, x) = 255;
        }
    }
    return t;
}

static grayImage templateCercle(int w, int h, float r) {
    grayImage t = makeGris(w, h, 0);
    int cx = w / 2;
    int cy = h / 2;

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            float dx = (float)(x - cx);
            float dy = (float)(y - cy);
            float v = std::sqrt(dx*dx + dy*dy);
            if (std::abs(v - r) < 1.5f) t.at(y, x) = 255;
        }
    }
    return t;
}

static RTable construireRTableDepuisTemplate(const grayImage& templ, int maxParBin, uint16_t seuilEdgeMag) {
    RTable lut;
    for (auto& v : lut) v.clear();

    ChampGradient tg = sobel(templ);
    int cx = templ.w / 2;
    int cy = templ.h / 2;

    for (int y = 1; y < templ.h - 1; ++y) {
        for (int x = 1; x < templ.w - 1; ++x) {
            if (tg.Mag(y, x) < seuilEdgeMag) continue;

            float gx = (float)tg.gradX[(size_t)y * (size_t)templ.w + (size_t)x];
            float gy = (float)tg.gradY[(size_t)y * (size_t)templ.w + (size_t)x];
            if (gx == 0.0f && gy == 0.0f) continue;

            int aBin = binDeg(std::atan2(gy, gx));

            Decalage off;
            off.dx = cx - x;
            off.dy = cy - y;

            if ((int)lut[aBin].size() < maxParBin) lut[aBin].push_back(off);
        }
    }
    return lut;
}

struct AccuImage {
    int w = 0, h = 0;
    std::vector<uint16_t> a;
    uint16_t& at(int y, int x) { return a[(size_t)y * (size_t)w + (size_t)x]; }
    uint16_t  at(int y, int x) const { return a[(size_t)y * (size_t)w + (size_t)x]; }
};

static AccuImage makeAccu(int w, int h) {
    AccuImage A;
    A.w = w; A.h = h;
    A.a.assign((size_t)w * (size_t)h, 0);
    return A;
}

static cv::Mat toMatAccu8(const AccuImage& A) {
    uint16_t maxV = 1;
    for (auto v : A.a) maxV = std::max<uint16_t>(maxV, v);

    cv::Mat m(A.h, A.w, CV_8UC1);
    for (int y = 0; y < A.h; ++y) {
        uint8_t* row = m.ptr<uint8_t>(y);
        for (int x = 0; x < A.w; ++x) {
            uint16_t v = A.at(y, x);
            row[x] = (uint8_t)((uint32_t)v * 255u / (uint32_t)maxV);
        }
    }
    return m;
}

// Vote
static void voter(AccuImage& accu, const grayImage& img, const ChampGradient& cg, const RTable& lut, uint16_t seuilEdgeMag) {
    for (int y = 1; y < img.h - 1; ++y) {
        for (int x = 1; x < img.w - 1; ++x) {
            uint16_t m = cg.Mag(y, x);
            if (m < seuilEdgeMag) continue;

            float gx = (float)cg.gradX[(size_t)y * (size_t)img.w + (size_t)x];
            float gy = (float)cg.gradY[(size_t)y * (size_t)img.w + (size_t)x];
            if (gx == 0.0f && gy == 0.0f) continue;

            int aBin = binDeg(std::atan2(gy, gx));
            const auto& offs = lut[aBin];

            for (const auto& off : offs) {
                int cx = x + off.dx;
                int cy = y + off.dy;
                if (cx < 0 || cx >= accu.w || cy < 0 || cy >= accu.h) continue;
                uint16_t& v = accu.at(cy, cx);
                if (v < std::numeric_limits<uint16_t>::max()) v++;
            }
        }
    }
}

struct PicBary {
    bool ok = false;
    int px = -1, py = -1;
    uint16_t peak = 0;
    float bx = 0.0f, by = 0.0f;
};

static PicBary barycentreLocalAutourMax(const AccuImage& A, int baryRadius) {
    PicBary out;
    uint16_t best = 0;
    int px = -1, py = -1;
    for (int y = 0; y < A.h; ++y) {
        for (int x = 0; x < A.w; ++x) {
            uint16_t v = A.at(y, x);
            if (v > best) { best = v; px = x; py = y; }
        }
    }
    if (px < 0 || py < 0) return out;

    double sw = 0.0;
    double sx = 0.0;
    double sy = 0.0;

    int x0 = std::max(0, px - baryRadius);
    int x1 = std::min(A.w - 1, px + baryRadius);
    int y0 = std::max(0, py - baryRadius);
    int y1 = std::min(A.h - 1, py + baryRadius);

    for (int y = y0; y <= y1; ++y) {
        for (int x = x0; x <= x1; ++x) {
            uint16_t w = A.at(y, x);
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

// TopK + NMS
struct PicPoint {
    int x = -1;
    int y = -1;
    uint16_t v = 0;
    float bx = 0.0f;
    float by = 0.0f;
};

static void nmsDisque(AccuImage& A, int cx, int cy, int r) {
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

static std::vector<PicPoint> topKpicsAvecBary(AccuImage A, int k, int nmsRadius, int baryRadius, uint16_t minVal) {
    std::vector<PicPoint> out;

    for (int i = 0; i < k; ++i) {
        PicBary b = barycentreLocalAutourMax(A, baryRadius);
        if (!b.ok) break;
        if (b.peak < minVal) break;

        PicPoint p;
        p.x = b.px; p.y = b.py; p.v = b.peak;
        p.bx = b.bx; p.by = b.by;
        out.push_back(p);

        nmsDisque(A, b.px, b.py, nmsRadius);
    }
    return out;
}

static bool choisirPaireYeux(
    const std::vector<PicPoint>& pics,
    int faceX, int faceY,
    int minDx, int maxDx, int maxDy,
    PicPoint& oeilGauche, PicPoint& oeilDroit
) {
    bool found = false;
    uint32_t bestSum = 0;

    for (size_t i = 0; i < pics.size(); ++i) {
        for (size_t j = i + 1; j < pics.size(); ++j) {
            int ax = (int)std::lround(pics[i].bx);
            int ay = (int)std::lround(pics[i].by);
            int bx = (int)std::lround(pics[j].bx);
            int by = (int)std::lround(pics[j].by);

            int dx = std::abs(ax - bx);
            int dy = std::abs(ay - by);

            if (dx < minDx) continue;
            if (dx > maxDx) continue;
            if (dy > maxDy) continue;

            // also constrain around face center (eyes should be above face center)
            if (ay > faceY) continue;
            if (by > faceY) continue;

            uint32_t sum = (uint32_t)pics[i].v + (uint32_t)pics[j].v;
            if (!found || sum > bestSum) {
                found = true;
                bestSum = sum;
                // left/right by x
                if (ax <= bx) { oeilGauche = pics[i]; oeilDroit = pics[j]; }
                else          { oeilGauche = pics[j]; oeilDroit = pics[i]; }
            }
        }
    }
    return found;
}

struct faceeyes {
    bool faceOk = false;
    int faceX = 0, faceY = 0;
    bool eyesOk = false;
    int ex1 = 0, ey1 = 0, ex2 = 0, ey2 = 0;
    int eyeR = 0;

    // debug
    ChampGradient dbgGrads;
    bool dbgFaceAccuOk = false;
    AccuImage dbgFaceAccu;
    bool dbgEyeAccuOk = false;
    AccuImage dbgEyeAccu;
};

struct facemodel { int rx = 0, ry = 0; RTable lut; };
struct eyemodel  { int r = 0; RTable lut; };

static faceeyes detectfaceeyes(
    const grayImage& img,
    const std::vector<facemodel>& faceModels,
    const std::vector<eyemodel>& eyeModels,
    uint16_t seuilFace, uint16_t seuilEye,
    uint16_t faceMinScore, uint16_t eyeMinPeak
) {
    faceeyes out;
    out.dbgGrads = sobel(img);

    // FACE: pick best model by peak
    uint16_t bestFacePeak = 0;
    int bestFaceX = 0, bestFaceY = 0;
    int bestRx = 0, bestRy = 0;
    AccuImage bestAccu = makeAccu(img.w, img.h);

    for (const auto& fm : faceModels) {
        AccuImage A = makeAccu(img.w, img.h);
        voter(A, img, out.dbgGrads, fm.lut, seuilFace);

        PicBary b = barycentreLocalAutourMax(A, 6);
        if (b.ok && b.peak >= bestFacePeak) {
            bestFacePeak = b.peak;
            bestFaceX = (int)std::lround(b.bx);
            bestFaceY = (int)std::lround(b.by);
            bestRx = fm.rx;
            bestRy = fm.ry;
            bestAccu = A;
        }
    }

    out.dbgFaceAccuOk = true;
    out.dbgFaceAccu = bestAccu;

    if (bestFacePeak < faceMinScore) {
        out.faceOk = false;
        return out;
    }

    out.faceOk = true;
    out.faceX = bestFaceX;
    out.faceY = bestFaceY;

    // EYES: define a search zone above face center (heuristic)
    int zx0 = clampInt(bestFaceX - bestRx, 0, img.w - 1);
    int zx1 = clampInt(bestFaceX + bestRx, 0, img.w - 1);
    int zy0 = clampInt(bestFaceY - bestRy, 0, img.h - 1);
    int zy1 = clampInt(bestFaceY, 0, img.h - 1);

    // Build sub-image zoneYeux
    int zw = std::max(1, zx1 - zx0 + 1);
    int zh = std::max(1, zy1 - zy0 + 1);
    grayImage zoneYeux;
    zoneYeux.w = zw;
    zoneYeux.h = zh;
    zoneYeux.p.assign((size_t)zw * (size_t)zh, 0);
    for (int y = 0; y < zh; ++y) {
        for (int x = 0; x < zw; ++x) {
            zoneYeux.at(y, x) = img.at(zy0 + y, zx0 + x);
        }
    }

    ChampGradient gradsYeux = sobel(zoneYeux);

    // for each radius model, pick best peak, keep global best
    uint16_t bestEyePeak = 0;
    int bestR = 0;
    AccuImage bestEyeAccu = makeAccu(zoneYeux.w, zoneYeux.h);
    std::vector<PicPoint> bestPics;

    for (const auto& em : eyeModels) {
        AccuImage A = makeAccu(zoneYeux.w, zoneYeux.h);
        voter(A, zoneYeux, gradsYeux, em.lut, seuilEye);

        auto pics = topKpicsAvecBary(A, /*k*/6, /*nmsRadius*/em.r * 2, /*baryRadius*/6, /*minVal*/eyeMinPeak);
        if (pics.empty()) continue;

        // score by strongest peak of this radius
        uint16_t localPeak = 0;
        for (auto& p : pics) localPeak = std::max<uint16_t>(localPeak, p.v);

        if (localPeak >= bestEyePeak) {
            bestEyePeak = localPeak;
            bestR = em.r;
            bestEyeAccu = A;
            bestPics = pics;
        }
    }

    out.dbgEyeAccuOk = true;
    out.dbgEyeAccu = bestEyeAccu;

    if (bestPics.empty()) {
        out.eyesOk = false;
        return out;
    }

    // pair selection constraints based on face size
    PicPoint og, od;
    int minDx = std::max(10, (int)std::lround(bestRx * 0.6));
    int maxDx = std::max(minDx + 10, (int)std::lround(bestRx * 1.6));
    int maxDy = std::max(10, (int)std::lround(bestRy * 0.35));

    bool pairOk = choisirPaireYeux(bestPics, bestFaceX - zx0, bestFaceY - zy0, minDx, maxDx, maxDy, og, od);
    if (!pairOk) {
        out.eyesOk = false;
        return out;
    }

    out.eyesOk = true;
    out.eyeR = bestR;

    // convert barycenters to global coords
    out.ex1 = zx0 + (int)std::lround(og.bx);
    out.ey1 = zy0 + (int)std::lround(og.by);
    out.ex2 = zx0 + (int)std::lround(od.bx);
    out.ey2 = zy0 + (int)std::lround(od.by);

    return out;
}

// MAIN
int main(int argc, char** argv) {
    bool doImage = false;
    std::string imagePath;

    // In --image mode, default is headless (no GUI windows, no blocking waitKey).
    // Use --gui to display debug windows and wait for a keypress.
    bool imageGui = false;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--image") {
            if (i + 1 < argc) { doImage = true; imagePath = argv[i + 1]; i++; }
            continue;
        }
        if (a == "--gui") {
            imageGui = true;
            continue;
        }
        if (a == "--no-gui" || a == "--headless") {
            imageGui = false;
            continue;
        }
    }

    const int EDGE_FACE = 140;
    const int EDGE_EYE  = 75;
    const uint16_t FACE_MIN_SCORE = 14;
    const uint16_t EYE_MIN_PEAK   = 5;

    std::vector<facemodel> faceModels;
    {
        const int scales[][2] = {
            {25, 45}, {30, 55}, {35, 65}, {45, 85}, {55, 105}, {65, 125}, {75, 145}
        };
        for (int i = 0; i < 7; ++i) {
            int rx = scales[i][0];
            int ry = scales[i][1];

            int tw = 2 * rx + 60;
            int th = 2 * ry + 60;

            grayImage t = templateEllipse(tw, th, (float)rx, (float)ry);
            RTable lut = construireRTableDepuisTemplate(t, 50, 220);

            facemodel fm;
            fm.rx = rx;
            fm.ry = ry;
            fm.lut = lut;
            faceModels.push_back(fm);
        }
    }

    std::vector<eyemodel> eyeModels;
    {
        for (int r = 6; r <= 18; r += 2) {
            int tw = 2 * r + 40;
            int th = 2 * r + 40;

            grayImage t = templateCercle(tw, th, (float)r);
            RTable lut = construireRTableDepuisTemplate(t, 40, 220);

            eyemodel em;
            em.r = r;
            em.lut = lut;
            eyeModels.push_back(em);
        }
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

        grayImage gray = bgrToGray(img);
        faceeyes r = detectfaceeyes(gray, faceModels, eyeModels, EDGE_FACE, EDGE_EYE, FACE_MIN_SCORE, EYE_MIN_PEAK);

        if (imageGui) {
            cv::imshow("Frame", img);
            cv::imshow("Gray", toMatGray8(gray));
            cv::imshow("Sobel", toMatMag8(r.dbgGrads));
            if (r.dbgFaceAccuOk) cv::imshow("Accu Face (best scale)", toMatAccu8(r.dbgFaceAccu));
            if (r.dbgEyeAccuOk)  cv::imshow("Accu Eyes (best radius)", toMatAccu8(r.dbgEyeAccu));
        }

        // IMPORTANT: stdout line for Python parser (GUI-independent)
        if (!r.faceOk) {
            std::cout << "Face=NOTFOUND\n";
        } else {
            std::cout << "Face=(" << r.faceX << "," << r.faceY << ")";
            if (!r.eyesOk) {
                std::cout << " Eyes=NOTFOUND\n";
            } else {
                std::cout << " Eyes=(" << r.ex1 << "," << r.ey1 << ")"
                        << " (" << r.ex2 << "," << r.ey2 << ")"
                        << " r=" << r.eyeR << "\n";
            }
        }
        std::cout.flush();

        if (imageGui) {
            std::cout << "Appuie sur une touche pour quitter.\n";
            std::cout.flush();
            cv::waitKey(0);
        }
        return 0;
    }


    // webcam
    cv::VideoCapture camera(0);
    if (!camera.isOpened()) {
        std::cerr << "Camera non ouverte.\n";
        return 1;
    }

    cv::Mat frame;
    camera >> frame;
    if (frame.empty()) {
        std::cerr << "Impossible de lire une frame.\n";
        return 1;
    }

    std::cout << "resolution camera : " << frame.cols << "x" << frame.rows << "\n";
    std::cout << "detection en cours (ESC pour quitter).\n";

    using Clock = std::chrono::steady_clock;
    auto lastOk = Clock::now() - std::chrono::seconds(5);
    const int cooldownOkSec = 5;

    int framecount = 0;

    while (true) {
        camera >> frame;
        if (frame.empty()) break;

        grayImage gray = bgrToGray(frame);
        faceeyes r = detectfaceeyes(gray, faceModels, eyeModels, EDGE_FACE, EDGE_EYE, FACE_MIN_SCORE, EYE_MIN_PEAK);

        // Example minimal live debug (optional)
        cv::imshow("Frame", frame);

        // Print detections (optional; useful for debugging)
        if (r.faceOk) {
            std::cout << "Face = (" << r.faceX << "," << r.faceY << ")";
            if (r.eyesOk) {
                std::cout << " Eyes = (" << r.ex1 << "," << r.ey1 << ")"
                          << " (" << r.ex2 << "," << r.ey2 << ")"
                          << " r = " << r.eyeR;
            } else {
                std::cout << " Eyes = NOTFOUND";
            }
            std::cout << "\n";
        } else {
            std::cout << "Face = NOTFOUND\n";
        }

        int k = cv::waitKey(1);
        if (k == 27) break; // ESC

        framecount++;
        (void)lastOk;
        (void)cooldownOkSec;
    }

    return 0;
}
