// FILE: vision/src/ght_face_eyes.cpp
#include <opencv2/videoio.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// -------------------- utils --------------------
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

static void showStep(const std::string& name, const cv::Mat& m, bool steps, int delayMs) {
    cv::imshow(name, m);
    if (steps) {
        cv::waitKey(0);
    } else if (delayMs > 0) {
        cv::waitKey(delayMs);
    }
}

// -------------------- image struct --------------------
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

static grayImage matToGrayImageU8(const cv::Mat& grayU8) {
    grayImage g;
    g.w = grayU8.cols;
    g.h = grayU8.rows;
    g.p.assign((size_t)g.w * (size_t)g.h, 0);
    for (int y = 0; y < g.h; ++y) {
        const uint8_t* row = grayU8.ptr<uint8_t>(y);
        for (int x = 0; x < g.w; ++x) g.at(y, x) = row[x];
    }
    return g;
}

static cv::Mat toMatGray8(const grayImage& g) {
    cv::Mat m(g.h, g.w, CV_8UC1);
    for (int y = 0; y < g.h; ++y) {
        uint8_t* row = m.ptr<uint8_t>(y);
        for (int x = 0; x < g.w; ++x) row[x] = g.at(y, x);
    }
    return m;
}

// -------------------- gradients --------------------
struct ChampGradient {
    int w = 0, h = 0;
    std::vector<uint16_t> mag; // magnitude
    std::vector<uint16_t> ang; // angle bins [0..359]

    uint16_t& m(int y, int x) { return mag[(size_t)y * (size_t)w + (size_t)x]; }
    uint16_t  m(int y, int x) const { return mag[(size_t)y * (size_t)w + (size_t)x]; }

    uint16_t& a(int y, int x) { return ang[(size_t)y * (size_t)w + (size_t)x]; }
    uint16_t  a(int y, int x) const { return ang[(size_t)y * (size_t)w + (size_t)x]; }
};

static ChampGradient sobel(const grayImage& img) {
    ChampGradient cg;
    cg.w = img.w;
    cg.h = img.h;
    cg.mag.assign((size_t)cg.w * (size_t)cg.h, 0);
    cg.ang.assign((size_t)cg.w * (size_t)cg.h, 0);

    auto at = [&](int y, int x) -> int {
        x = clampInt(x, 0, img.w - 1);
        y = clampInt(y, 0, img.h - 1);
        return (int)img.at(y, x);
    };

    for (int y = 0; y < img.h; ++y) {
        for (int x = 0; x < img.w; ++x) {
            int gx =
                -1 * at(y - 1, x - 1) + 1 * at(y - 1, x + 1) +
                -2 * at(y,     x - 1) + 2 * at(y,     x + 1) +
                -1 * at(y + 1, x - 1) + 1 * at(y + 1, x + 1);

            int gy =
                -1 * at(y - 1, x - 1) + -2 * at(y - 1, x) + -1 * at(y - 1, x + 1) +
                 1 * at(y + 1, x - 1) +  2 * at(y + 1, x) +  1 * at(y + 1, x + 1);

            float mag = std::sqrt((float)gx * (float)gx + (float)gy * (float)gy);
            float ang = std::atan2((float)gy, (float)gx);

            int im = (int)std::lround(mag);
            cg.m(y, x) = (uint16_t)clampInt(im, 0, 65535);
            cg.a(y, x) = (uint16_t)binDeg(ang);
        }
    }
    return cg;
}

// For GUI: normalize magnitude to [0..255] by min/max (readable even when edges are weak)
static cv::Mat toMatMag8_norm(const ChampGradient& cg) {
    cv::Mat m(cg.h, cg.w, CV_8UC1);

    uint16_t minv = std::numeric_limits<uint16_t>::max();
    uint16_t maxv = 0;
    for (int y = 0; y < cg.h; ++y) {
        for (int x = 0; x < cg.w; ++x) {
            uint16_t v = cg.m(y, x);
            minv = std::min<uint16_t>(minv, v);
            maxv = std::max<uint16_t>(maxv, v);
        }
    }
    if (maxv <= minv) {
        m.setTo(0);
        return m;
    }

    for (int y = 0; y < cg.h; ++y) {
        uint8_t* row = m.ptr<uint8_t>(y);
        for (int x = 0; x < cg.w; ++x) {
            float f = (float)(cg.m(y, x) - minv) / (float)(maxv - minv);
            row[x] = (uint8_t)clampInt((int)std::lround(255.0f * f), 0, 255);
        }
    }
    return m;
}

// -------------------- accumulator + R-Table --------------------
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

struct RTable {
    // angle bin -> list of (dx, dy)
    std::array<std::vector<std::pair<int16_t, int16_t>>, 360> lut;
};

static void voter(
    AccuImage& A,
    const grayImage& img,
    const ChampGradient& grads,
    const RTable& rtable,
    uint16_t seuilMag
) {
    // vote for all pixels with sufficient gradient magnitude
    for (int y = 0; y < img.h; ++y) {
        for (int x = 0; x < img.w; ++x) {
            uint16_t mag = grads.m(y, x);
            if (mag < seuilMag) continue;
            uint16_t ang = grads.a(y, x);
            const auto& vec = rtable.lut[(size_t)ang];
            if (vec.empty()) continue;

            for (const auto& d : vec) {
                int cx = x + d.first;
                int cy = y + d.second;
                if (cx < 0 || cy < 0 || cx >= A.w || cy >= A.h) continue;
                uint16_t& cell = A.at(cy, cx);
                if (cell < 65535) cell++;
            }
        }
    }
}

struct PicBary {
    bool ok = false;
    float bx = 0.0f, by = 0.0f;
    uint16_t peak = 0;
};

static PicBary barycentreLocalAutourMax(const AccuImage& A, int radius) {
    // find max
    uint16_t peak = 0;
    int px = 0, py = 0;
    for (int y = 0; y < A.h; ++y) {
        for (int x = 0; x < A.w; ++x) {
            uint16_t v = A.at(y, x);
            if (v >= peak) { peak = v; px = x; py = y; }
        }
    }
    if (peak == 0) return PicBary{false, 0, 0, 0};

    int x0 = clampInt(px - radius, 0, A.w - 1);
    int x1 = clampInt(px + radius, 0, A.w - 1);
    int y0 = clampInt(py - radius, 0, A.h - 1);
    int y1 = clampInt(py + radius, 0, A.h - 1);

    double sum = 0.0;
    double sx = 0.0, sy = 0.0;

    for (int y = y0; y <= y1; ++y) {
        for (int x = x0; x <= x1; ++x) {
            double w = (double)A.at(y, x);
            sum += w;
            sx += w * (double)x;
            sy += w * (double)y;
        }
    }

    if (sum <= 0.0) return PicBary{false, 0, 0, peak};
    return PicBary{true, (float)(sx / sum), (float)(sy / sum), peak};
}

struct PicPoint {
    int x = 0, y = 0;
    float bx = 0.0f, by = 0.0f;
    uint16_t v = 0;
};

static std::vector<PicPoint> topKpicsAvecBary(
    const AccuImage& A,
    int k,
    int nmsRadius,
    int baryRadius,
    uint16_t minVal
) {
    // naive: take all candidates above minVal, sort desc, apply NMS, compute barycenter
    struct Cand { int x,y; uint16_t v; };
    std::vector<Cand> cands;
    cands.reserve(2048);

    for (int y = 0; y < A.h; ++y) {
        for (int x = 0; x < A.w; ++x) {
            uint16_t v = A.at(y, x);
            if (v >= minVal) cands.push_back({x,y,v});
        }
    }
    std::sort(cands.begin(), cands.end(), [](const Cand& a, const Cand& b){ return a.v > b.v; });

    std::vector<PicPoint> out;
    for (const auto& c : cands) {
        bool tooClose = false;
        for (const auto& p : out) {
            int dx = c.x - p.x;
            int dy = c.y - p.y;
            if (dx*dx + dy*dy <= nmsRadius*nmsRadius) { tooClose = true; break; }
        }
        if (tooClose) continue;

        // barycenter around (c.x,c.y)
        int x0 = clampInt(c.x - baryRadius, 0, A.w - 1);
        int x1 = clampInt(c.x + baryRadius, 0, A.w - 1);
        int y0 = clampInt(c.y - baryRadius, 0, A.h - 1);
        int y1 = clampInt(c.y + baryRadius, 0, A.h - 1);

        double sum = 0.0, sx = 0.0, sy = 0.0;
        for (int y = y0; y <= y1; ++y) {
            for (int x = x0; x <= x1; ++x) {
                double w = (double)A.at(y, x);
                sum += w;
                sx += w * (double)x;
                sy += w * (double)y;
            }
        }
        PicPoint pp;
        pp.x = c.x; pp.y = c.y; pp.v = c.v;
        if (sum > 0.0) { pp.bx = (float)(sx / sum); pp.by = (float)(sy / sum); }
        else { pp.bx = (float)c.x; pp.by = (float)c.y; }
        out.push_back(pp);

        if ((int)out.size() >= k) break;
    }
    return out;
}

// pair selection
static bool choisirPaireYeux(
    const std::vector<PicPoint>& pics,
    int faceCxInZone,
    int faceCyInZone,
    int minDx, int maxDx,
    int maxDy,
    PicPoint& oeilGauche,
    PicPoint& oeilDroit
) {
    bool found = false;
    uint32_t best = 0;

    for (size_t i = 0; i < pics.size(); ++i) {
        for (size_t j = i + 1; j < pics.size(); ++j) {
            const auto& p1 = pics[i];
            const auto& p2 = pics[j];

            // order left-right
            const auto& L = (p1.bx <= p2.bx) ? p1 : p2;
            const auto& R = (p1.bx <= p2.bx) ? p2 : p1;

            int dx = (int)std::lround(R.bx - L.bx);
            int dy = (int)std::lround(std::fabs(R.by - L.by));

            if (dx < minDx || dx > maxDx) continue;
            if (dy > maxDy) continue;

            // keep roughly above face center
            if ((int)std::lround(L.by) > faceCyInZone) continue;
            if ((int)std::lround(R.by) > faceCyInZone) continue;

            uint32_t score = (uint32_t)L.v + (uint32_t)R.v;
            if (!found || score > best) {
                found = true;
                best = score;
                oeilGauche = L;
                oeilDroit = R;
            }
        }
    }
    return found;
}

// -------------------- templates --------------------
static grayImage templateEllipse(int w, int h, float rx, float ry) {
    grayImage img = makeGris(w, h, 255);
    int cx = w / 2;
    int cy = h / 2;
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

static grayImage templateCercle(int w, int h, float r) {
    grayImage img = makeGris(w, h, 255);
    int cx = w / 2;
    int cy = h / 2;
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            float dx = (float)(x - cx);
            float dy = (float)(y - cy);
            float d = std::sqrt(dx*dx + dy*dy);
            if (std::fabs(d - r) < 2.5f) img.at(y, x) = 0;
        }
    }
    return img;
}

static RTable construireRTableDepuisTemplate(const grayImage& templ, uint16_t minMag, uint16_t maxMag) {
    ChampGradient g = sobel(templ);

    // template center
    int cx = templ.w / 2;
    int cy = templ.h / 2;

    RTable rt;

    for (int y = 0; y < templ.h; ++y) {
        for (int x = 0; x < templ.w; ++x) {
            uint16_t mag = g.m(y, x);
            if (mag < minMag || mag > maxMag) continue;
            uint16_t ang = g.a(y, x);

            int dx = cx - x;
            int dy = cy - y;
            rt.lut[(size_t)ang].push_back({(int16_t)dx, (int16_t)dy});
        }
    }
    return rt;
}

// -------------------- models --------------------
struct facemodel { int rx = 0, ry = 0; RTable lut; };
struct eyemodel  { int r = 0; RTable lut; };

// -------------------- adaptive threshold helper --------------------
static uint16_t magPercentile(const ChampGradient& cg, double q /*0..1*/) {
    // sample to reduce cost
    std::vector<uint16_t> s;
    s.reserve((size_t)(cg.w * cg.h / 4));
    for (int y = 0; y < cg.h; y += 2) {
        for (int x = 0; x < cg.w; x += 2) {
            s.push_back(cg.m(y, x));
        }
    }
    if (s.empty()) return 0;

    size_t idx = (size_t)std::lround(q * (double)(s.size() - 1));
    idx = std::min(idx, s.size() - 1);

    std::nth_element(s.begin(), s.begin() + idx, s.end());
    return s[idx];
}

struct faceeyes {
    bool faceOk = false;
    int faceX = 0, faceY = 0;
    int faceRx = 0, faceRy = 0;

    int eyeRoiX = 0, eyeRoiY = 0, eyeRoiW = 0, eyeRoiH = 0;

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

static faceeyes detectfaceeyes(
    const grayImage& img,
    const std::vector<facemodel>& faceModels,
    const std::vector<eyemodel>& eyeModels,
    uint16_t seuilFace, uint16_t seuilEye,
    uint16_t faceMinScore, uint16_t eyeMinPeak
) {
    faceeyes out;
    out.dbgGrads = sobel(img);

    // FACE: pick best model by peak (barycentered max)
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
    out.faceRx = bestRx;
    out.faceRy = bestRy;

    // EYES: ROI above face center (tighten to reduce window edges)
    int zx0 = clampInt(bestFaceX - (int)std::lround(bestRx * 1.2), 0, img.w - 1);
    int zx1 = clampInt(bestFaceX + (int)std::lround(bestRx * 1.2), 0, img.w - 1);
    int zy0 = clampInt(bestFaceY - (int)std::lround(bestRy * 1.1), 0, img.h - 1);
    int zy1 = clampInt(bestFaceY - (int)std::lround(bestRy * 0.15), 0, img.h - 1);

    if (zx1 <= zx0 || zy1 <= zy0) {
        out.eyesOk = false;
        return out;
    }

    out.eyeRoiX = zx0;
    out.eyeRoiY = zy0;
    out.eyeRoiW = (zx1 - zx0 + 1);
    out.eyeRoiH = (zy1 - zy0 + 1);

    // Build sub-image zoneYeux
    grayImage zoneYeux;
    zoneYeux.w = out.eyeRoiW;
    zoneYeux.h = out.eyeRoiH;
    zoneYeux.p.assign((size_t)zoneYeux.w * (size_t)zoneYeux.h, 0);
    for (int y = 0; y < zoneYeux.h; ++y) {
        for (int x = 0; x < zoneYeux.w; ++x) {
            zoneYeux.at(y, x) = img.at(zy0 + y, zx0 + x);
        }
    }

    ChampGradient gradsYeux = sobel(zoneYeux);

    // for each radius model, pick best peaks list, keep global best
    uint16_t bestEyePeak = 0;
    int bestR = 0;
    AccuImage bestEyeAccu = makeAccu(zoneYeux.w, zoneYeux.h);
    std::vector<PicPoint> bestPics;

    for (const auto& em : eyeModels) {
        AccuImage A = makeAccu(zoneYeux.w, zoneYeux.h);
        voter(A, zoneYeux, gradsYeux, em.lut, seuilEye);

        auto pics = topKpicsAvecBary(A, /*k*/6, /*nmsRadius*/em.r * 2, /*baryRadius*/6, /*minVal*/eyeMinPeak);
        if (pics.empty()) continue;

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
    int minDx = std::max(10, (int)std::lround(bestRx * 0.55));
    int maxDx = std::max(minDx + 10, (int)std::lround(bestRx * 1.60));
    int maxDy = std::max(10, (int)std::lround(bestRy * 0.30));

    bool pairOk = choisirPaireYeux(bestPics, bestFaceX - zx0, bestFaceY - zy0, minDx, maxDx, maxDy, og, od);
    if (!pairOk) {
        out.eyesOk = false;
        return out;
    }

    out.eyesOk = true;
    out.eyeR = bestR;

    out.ex1 = zx0 + (int)std::lround(og.bx);
    out.ey1 = zy0 + (int)std::lround(og.by);
    out.ex2 = zx0 + (int)std::lround(od.bx);
    out.ey2 = zy0 + (int)std::lround(od.by);

    return out;
}

static cv::Mat toMatAccu8(const AccuImage& A) {
    cv::Mat m(A.h, A.w, CV_8UC1);
    uint16_t maxv = 1;
    for (int y = 0; y < A.h; ++y)
        for (int x = 0; x < A.w; ++x)
            maxv = std::max<uint16_t>(maxv, A.at(y, x));

    for (int y = 0; y < A.h; ++y) {
        uint8_t* row = m.ptr<uint8_t>(y);
        for (int x = 0; x < A.w; ++x) {
            float f = (float)A.at(y, x) / (float)maxv;
            row[x] = (uint8_t)clampInt((int)std::lround(255.0f * f), 0, 255);
        }
    }
    return m;
}

static void drawOverlay(cv::Mat& frame, const faceeyes& r) {
    if (r.faceOk) {
        cv::circle(frame, cv::Point(r.faceX, r.faceY), 6, cv::Scalar(0, 255, 0), 2);
    }
    if (r.eyeRoiW > 0 && r.eyeRoiH > 0) {
        cv::rectangle(frame, cv::Rect(r.eyeRoiX, r.eyeRoiY, r.eyeRoiW, r.eyeRoiH), cv::Scalar(255, 0, 0), 2);
    }
    if (r.eyesOk) {
        cv::circle(frame, cv::Point(r.ex1, r.ey1), r.eyeR, cv::Scalar(0, 255, 255), 2);
        cv::circle(frame, cv::Point(r.ex2, r.ey2), r.eyeR, cv::Scalar(0, 255, 255), 2);
    }
}

int main(int argc, char** argv) {
    bool doImage = false;
    std::string imagePath;

    // GUI controls (kept compatible with your current code)
    bool imageGui = false;
    bool guiSteps = false;
    int guiDelayMs = 0;

    // New options
    bool useEqHist = true;
    bool useClahe = false;
    int blurK = 5;               // odd, 0 disables
    bool autoThr = true;
    int faceEdgeUser = -1;
    int eyeEdgeUser  = -1;
    int faceMinUser  = -1;
    int eyeMinUser   = -1;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];

        if (a == "--image") {
            if (i + 1 < argc) { doImage = true; imagePath = argv[i + 1]; i++; }
            continue;
        }

        if (a == "--gui") { imageGui = true; continue; }
        if (a == "--gui-steps") { imageGui = true; guiSteps = true; continue; }
        if (a == "--gui-delay-ms") {
            if (i + 1 < argc) { imageGui = true; guiDelayMs = std::max(0, std::atoi(argv[i + 1])); i++; }
            continue;
        }
        if (a == "--no-gui" || a == "--headless") { imageGui = false; guiSteps = false; guiDelayMs = 0; continue; }

        // New args
        if (a == "--no-eq") { useEqHist = false; continue; }
        if (a == "--clahe") { useClahe = true; continue; }
        if (a == "--blur") {
            if (i + 1 < argc) { blurK = std::atoi(argv[i + 1]); i++; }
            continue;
        }
        if (a == "--no-auto-threshold") { autoThr = false; continue; }
        if (a == "--face-edge") {
            if (i + 1 < argc) { faceEdgeUser = std::atoi(argv[i + 1]); i++; }
            continue;
        }
        if (a == "--eye-edge") {
            if (i + 1 < argc) { eyeEdgeUser = std::atoi(argv[i + 1]); i++; }
            continue;
        }
        if (a == "--face-min-score") {
            if (i + 1 < argc) { faceMinUser = std::atoi(argv[i + 1]); i++; }
            continue;
        }
        if (a == "--eye-min-peak") {
            if (i + 1 < argc) { eyeMinUser = std::atoi(argv[i + 1]); i++; }
            continue;
        }
    }

    // Defaults (same as your current baseline, but we will override if auto-threshold)
    uint16_t EDGE_FACE = 140;
    uint16_t EDGE_EYE  = 75;
    uint16_t FACE_MIN_SCORE = 14;
    uint16_t EYE_MIN_PEAK   = 5;

    if (faceEdgeUser >= 0) EDGE_FACE = (uint16_t)clampInt(faceEdgeUser, 0, 65535);
    if (eyeEdgeUser  >= 0) EDGE_EYE  = (uint16_t)clampInt(eyeEdgeUser, 0, 65535);
    if (faceMinUser  >= 0) FACE_MIN_SCORE = (uint16_t)clampInt(faceMinUser, 0, 65535);
    if (eyeMinUser   >= 0) EYE_MIN_PEAK   = (uint16_t)clampInt(eyeMinUser, 0, 65535);

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
            fm.rx = rx; fm.ry = ry; fm.lut = lut;
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
            em.r = r; em.lut = lut;
            eyeModels.push_back(em);
        }
    }

    if (!doImage) {
        std::cerr << "Usage: ght_face_eyes --image <path> [--gui|--no-gui] [--gui-steps] [--gui-delay-ms N]\n"
                  << "  Options:\n"
                  << "    --no-eq                 : disable histogram equalization\n"
                  << "    --clahe                 : use CLAHE instead of equalizeHist\n"
                  << "    --blur <oddK>           : gaussian blur kernel (odd). 0 disables. default=5\n"
                  << "    --no-auto-threshold     : use fixed EDGE_* constants\n"
                  << "    --face-edge <v>         : override EDGE_FACE\n"
                  << "    --eye-edge <v>          : override EDGE_EYE\n"
                  << "    --face-min-score <v>    : override FACE_MIN_SCORE\n"
                  << "    --eye-min-peak <v>      : override EYE_MIN_PEAK\n";
        return 2;
    }

    cv::Mat bgr = cv::imread(imagePath);
    if (bgr.empty()) {
        std::cerr << "Erreur: impossible de lire l'image: " << imagePath << "\n";
        return 1;
    }
    if (bgr.channels() != 3) {
        std::cerr << "Erreur: image doit etre en BGR (3 canaux)\n";
        return 1;
    }

    // Preprocess
    cv::Mat gray;
    cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);

    if (useClahe) {
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
        clahe->apply(gray, gray);
    } else if (useEqHist) {
        cv::equalizeHist(gray, gray);
    }

    if (blurK > 0) {
        if (blurK % 2 == 0) blurK += 1;
        blurK = std::max(1, blurK);
        cv::GaussianBlur(gray, gray, cv::Size(blurK, blurK), 0.0);
    }

    grayImage g = matToGrayImageU8(gray);

    // Auto thresholds based on gradient percentiles
    if (autoThr && faceEdgeUser < 0 && eyeEdgeUser < 0) {
        ChampGradient cg = sobel(g);
        // These heuristics are designed to prevent "no votes" on low-contrast frames.
        // p90 tends to be "strong edges"; we pick fractions for face/eyes.
        uint16_t p90 = magPercentile(cg, 0.90);
        uint16_t p80 = magPercentile(cg, 0.80);

        // guard rails
        uint16_t faceT = (uint16_t)clampInt((int)std::lround((double)p90 * 0.70), 20, 600);
        uint16_t eyeT  = (uint16_t)clampInt((int)std::lround((double)p80 * 0.55), 15, 500);

        EDGE_FACE = faceT;
        EDGE_EYE  = eyeT;
    }

    faceeyes r = detectfaceeyes(g, faceModels, eyeModels, EDGE_FACE, EDGE_EYE, FACE_MIN_SCORE, EYE_MIN_PEAK);

    // Print result (keep parser-compatible format)
    if (!r.faceOk) {
        std::cout << "Face=NOTFOUND\n";
    } else {
        std::cout << "Face=(" << r.faceX << "," << r.faceY << ")\n";
    }

    if (!r.eyesOk) {
        std::cout << "Eyes=NOTFOUND\n";
    } else {
        std::cout << "Eyes=(" << r.ex1 << "," << r.ey1 << ") (" << r.ex2 << "," << r.ey2 << ") r=" << r.eyeR << "\n";
    }

    // Also print debug thresholds to stderr (doesn't break stdout parser)
    std::cerr << "[DBG] EDGE_FACE=" << EDGE_FACE
              << " EDGE_EYE=" << EDGE_EYE
              << " FACE_MIN_SCORE=" << FACE_MIN_SCORE
              << " EYE_MIN_PEAK=" << EYE_MIN_PEAK
              << " autoThr=" << (autoThr ? "1" : "0")
              << " eq=" << (useEqHist ? "1" : "0")
              << " clahe=" << (useClahe ? "1" : "0")
              << " blurK=" << blurK
              << "\n";

    if (imageGui) {
        cv::Mat overlay = bgr.clone();
        drawOverlay(overlay, r);

        showStep("Frame", overlay, guiSteps, guiDelayMs);
        showStep("Gray(pre)", gray, guiSteps, guiDelayMs);
        showStep("Sobel(norm)", toMatMag8_norm(r.dbgGrads), guiSteps, guiDelayMs);
        if (r.dbgFaceAccuOk) showStep("Accu Face (best scale)", toMatAccu8(r.dbgFaceAccu), guiSteps, guiDelayMs);
        if (r.dbgEyeAccuOk)  showStep("Accu Eyes (best radius)", toMatAccu8(r.dbgEyeAccu), guiSteps, guiDelayMs);

        if (!guiSteps && guiDelayMs <= 0) {
            // keep same behavior as your existing tool: block on a key when GUI is enabled without delay/steps
            cv::waitKey(0);
        }
    }

    return 0;
}
