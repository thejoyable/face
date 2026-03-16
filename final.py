import streamlit as st
import torch
from torch import nn
from torchvision import models, transforms
import numpy as np
import cv2
import dlib
import stone
import tempfile
import os
from PIL import Image
from collections import OrderedDict


# ═══════════════════════════════════════════════════════════════
#  COMPACT CSS
# ═══════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Skin Tone Detector · Dual Path",
    page_icon="🎨",
    layout="wide",
)

st.markdown("""
<style>
    /* tighten block spacing */
    .block-container { padding-top: 1.5rem; padding-bottom: 1rem; }
    /* smaller image captions */
    .stImage > div > div > p { font-size: 0.75rem !important; margin: 0 !important; }
    /* smaller subheaders */
    h2 { font-size: 1.25rem !important; margin-top: 0.5rem !important; }
    h3 { font-size: 1.05rem !important; margin-top: 0.3rem !important; }
    /* reduce metric padding */
    [data-testid="stMetric"] { padding: 0.3rem 0 !important; }
    [data-testid="stMetricValue"] { font-size: 1.1rem !important; }
    [data-testid="stMetricLabel"] { font-size: 0.75rem !important; }
    /* tighter dividers */
    hr { margin: 0.5rem 0 !important; }
    /* tighter expanders */
    .streamlit-expanderHeader { font-size: 0.85rem !important; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#  0. DLIB FACE DETECTOR
# ═══════════════════════════════════════════════════════════════

@st.cache_resource
def load_face_detector():
    return dlib.get_frontal_face_detector()


def detect_and_crop_face(image_rgb, detector, padding=30):
    faces = detector(image_rgb, 1)
    if len(faces) == 0:
        return None, None, []
    face = max(faces, key=lambda r: (r.right() - r.left()) * (r.bottom() - r.top()))
    h, w = image_rgb.shape[:2]
    y1 = max(0, face.top()  - padding)
    x1 = max(0, face.left() - padding)
    y2 = min(h, face.bottom() + padding)
    x2 = min(w, face.right()  + padding)
    return image_rgb[y1:y2, x1:x2].copy(), (y1, y2, x1, x2), faces


# ═══════════════════════════════════════════════════════════════
#  1. LINKNET34 MODEL
# ═══════════════════════════════════════════════════════════════

class DecoderBlockLinkNet(nn.Module):
    def __init__(self, in_channels, n_filters):
        super().__init__()
        self.relu  = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.deconv2 = nn.ConvTranspose2d(
            in_channels // 4, in_channels // 4, 4, stride=2, padding=1
        )
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)

    def forward(self, x):
        x = self.relu(self.norm1(self.conv1(x)))
        x = self.relu(self.norm2(self.deconv2(x)))
        x = self.relu(self.norm3(self.conv3(x)))
        return x


class LinkNet34(nn.Module):
    # FIX #1: removed deprecated `pretrained` parameter
    def __init__(self, num_classes=1, num_channels=3, use_pretrained=False):
        super().__init__()
        filters = [64, 128, 256, 512]
        # FIX: use `weights=` instead of `pretrained=`
        resnet = models.resnet34(
            weights=models.ResNet34_Weights.DEFAULT if use_pretrained else None
        )
        self.firstconv    = resnet.conv1
        self.firstbn      = resnet.bn1
        self.firstrelu    = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4
        self.decoder4 = DecoderBlockLinkNet(filters[3], filters[2])
        self.decoder3 = DecoderBlockLinkNet(filters[2], filters[1])
        self.decoder2 = DecoderBlockLinkNet(filters[1], filters[0])
        self.decoder1 = DecoderBlockLinkNet(filters[0], filters[0])
        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 3, stride=2)
        self.finalrelu1   = nn.ReLU(inplace=True)
        self.finalconv2   = nn.Conv2d(32, 32, 3)
        self.finalrelu2   = nn.ReLU(inplace=True)
        self.finalconv3   = nn.Conv2d(32, num_classes, 2, padding=1)

    def forward(self, x):
        x  = self.firstconv(x);    x  = self.firstbn(x)
        x  = self.firstrelu(x);    x  = self.firstmaxpool(x)
        e1 = self.encoder1(x);     e2 = self.encoder2(e1)
        e3 = self.encoder3(e2);    e4 = self.encoder4(e3)
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)
        f  = self.finalrelu1(self.finaldeconv1(d1))
        f  = self.finalrelu2(self.finalconv2(f))
        return self.finalconv3(f)


@st.cache_resource
def load_linknet():
    model = LinkNet34(use_pretrained=False)
    model.load_state_dict(torch.load("linknet.pth", map_location="cpu"))
    model.eval()
    return model


# ═══════════════════════════════════════════════════════════════
#  2. YCrCb / HSV THRESHOLDING
# ═══════════════════════════════════════════════════════════════

def skin_threshold_ycrcb(bgr):
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    return cv2.inRange(ycrcb, (0, 133, 77), (255, 173, 127))


def skin_threshold_hsv(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, (0, 20, 70), (20, 255, 255))


def refine_mask(mask):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.erode(mask, k, iterations=1)
    mask = cv2.dilate(mask, k, iterations=2)
    mask = cv2.GaussianBlur(mask, (3, 3), 0)
    _, mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)
    return mask


# ═══════════════════════════════════════════════════════════════
#  3A. PATH 1 — K-Means + Custom Palette
# ═══════════════════════════════════════════════════════════════

TONE_PALETTE = OrderedDict([
    ("Fair",  (245, 224, 210)),
    ("Fair",  (228, 196, 168)),
    ("Fair",  (209, 173, 141)),
    ("Fair",  (182, 142, 108)),
    ("Brown", (176, 128,  98)),
    ("brown",  (173, 121,  93)),
    ("Brown",  (170, 113,  87)),
    ("Brown", (155, 118,  83)),
    ("Brown", (138, 101,  71)),
    ("Dark",  (120,  85,  58)),
    ("Dark",  ( 95,  65,  60)),
    ("Dark",  ( 51,  33,  22)),
])

def classify_tone(avg_rgb):
    avg = np.array(avg_rgb, dtype=np.float64)
    best_label, best_dist, best_rgb = None, float("inf"), None
    for label, ref in TONE_PALETTE.items():
        d = np.linalg.norm(avg - np.array(ref, np.float64))
        if d < best_dist:
            best_label, best_dist, best_rgb = label, d, ref
    pal_hex  = "#{:02X}{:02X}{:02X}".format(*best_rgb)
    your_hex = "#{:02X}{:02X}{:02X}".format(
        int(avg_rgb[0]), int(avg_rgb[1]), int(avg_rgb[2])
    )
    return best_label, pal_hex, your_hex, best_dist


def dominant_colors_kmeans(pixels, k=3):
    if len(pixels) < k:
        return [], []
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.2)
    _, labels, centres = cv2.kmeans(
        pixels.astype(np.float32), k, None, crit, 5, cv2.KMEANS_PP_CENTERS
    )
    counts = np.bincount(labels.flatten())
    order  = np.argsort(-counts)
    return (
        [tuple(centres[i].astype(int)) for i in order],
        [counts[i] for i in order],
    )


# ═══════════════════════════════════════════════════════════════
#  3B. PATH 2 — Stone Library
# ═══════════════════════════════════════════════════════════════

STONE_PALETTE = [
    "#373028",  # Deep Dark
    "#422811",  # Dark Espresso
    "#513B2E",  # Dark Brown

    "#6F503C",  # Medium Brown
    "#81654F",  # Warm Brown
    "#9D7A54",  # Light Brown

    "#BEA07E",  # Tan

    "#E5C8A6",  # Light Tan
    "#E7C1B8",  # Fair Beige
    "#F3DAD6",  # Light Fair
    "#FBF2F3",  # Very Fair
]
STONE_LABELS = [
    "Deep Dark",
    "Dark Espresso",
    "Dark Brown",
    "Medium Brown",
    "Warm Brown",
    "Light Brown",
    "Tan",
    "Light Tan",
    "Fair Beige",
    "Light Fair",
    "Very Fair"
]

def run_stone(image_rgb):
    """Save RGB image → temp file → stone.process → results dict."""
    fd, tmp = tempfile.mkstemp(suffix=".jpg")
    os.close(fd)
    try:
        Image.fromarray(image_rgb).save(tmp)
        result = stone.process(
            tmp,
            image_type="color",
            tone_palette=STONE_PALETTE,
            tone_labels=STONE_LABELS,
            return_report_image=True,
        )
        reports = result.pop("report_images", {})
        return result, reports
    except Exception as exc:
        return None, str(exc)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


# ═══════════════════════════════════════════════════════════════
#  4. UI HELPER
# ═══════════════════════════════════════════════════════════════

def swatch(hex_color, text, h=50):
    return (
        f'<div style="background:{hex_color};width:100%;height:{h}px;'
        f'border-radius:8px;border:2px solid #555;display:flex;'
        f'align-items:center;justify-content:center;color:#fff;'
        f'font-weight:600;font-size:0.8rem;text-shadow:1px 1px 2px #000">'
        f'{text}</div>'
    )


# ═══════════════════════════════════════════════════════════════
#  5. STREAMLIT APP
# ═══════════════════════════════════════════════════════════════

st.title("🎨 Skin Tone Detector — Dual Path")
st.caption(
    "**Shared →** dlib HOG · LinkNet34 · YCrCb/HSV  "
    "**Then →** 🎯 K-Means palette  •  🪨 Stone classifier"
)

# ── sidebar ──────────────────────────────────────────────────
st.sidebar.header("⚙️ Settings")
face_pad   = st.sidebar.slider("Face crop padding (px)", 0, 80, 30, 5)
seg_thresh = st.sidebar.slider("Seg threshold", 0.1, 0.9, 0.5, 0.05)
cspace     = st.sidebar.radio(
    "Colour-space (Stage 2)", ["ycrcb", "hsv"],
    index=1,
    format_func=lambda m: "YCrCb (recommended)" if m == "ycrcb" else "HSV",
)
n_k = st.sidebar.slider("K-Means K (Path 1)", 1, 5, 3)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Pipeline**\n\n"
    "```\n"
    "Upload → dlib crop → LinkNet34\n"
    "→ skin mask → K-Means / Stone\n"
    "```"
)

# ── upload ───────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Upload a face image", type=["jpg", "jpeg", "png", "webp"]
)

if uploaded is not None:

    image   = Image.open(uploaded).convert("RGB")
    orig_np = np.array(image)

    # Show upload small — in a column so it doesn't stretch full width
    up_col1, up_col2 = st.columns([1, 2])
    up_col1.image(image, caption="Uploaded", width=400)

    if True:

        # ═════════════════════════════════════════════════════
        #  STAGE 0 — dlib HOG
        # ═════════════════════════════════════════════════════
        with st.spinner("Stage 0 — dlib face detection …"):
            detector = load_face_detector()
            crop_rgb, bbox, all_faces = detect_and_crop_face(
                orig_np, detector, padding=face_pad
            )

        if crop_rgb is None:
            st.error("❌ No face detected by dlib. Try a clearer photo.")
            st.stop()

        y1, y2, x1, x2 = bbox

        st.divider()
        with st.expander(
            f"👤 Stage 0 — dlib ({len(all_faces)} face"
            f"{'s' if len(all_faces) != 1 else ''})",
            expanded=True,
        ):
            s0a, s0b = st.columns(2)
            boxed = orig_np.copy()
            for f in all_faces:
                cv2.rectangle(
                    boxed, (f.left(), f.top()), (f.right(), f.bottom()),
                    (0, 255, 0), 3,
                )
            cv2.rectangle(boxed, (x1, y1), (x2, y2), (255, 0, 0), 3)
            s0a.image(boxed,    caption="Detected (blue=selected)", width=200)
            s0b.image(crop_rgb, caption=f"Crop {x2-x1}×{y2-y1}",   width=200)

        # ═════════════════════════════════════════════════════
        #  STAGE 1 — LinkNet34 segmentation
        # ═════════════════════════════════════════════════════
        with st.spinner("Stage 1 — LinkNet34 segmentation …"):
            model = load_linknet()
            ch, cw = crop_rgb.shape[:2]
            tfm = transforms.Compose([
                transforms.Resize((256, 256)), transforms.ToTensor()
            ])
            inp = tfm(Image.fromarray(crop_rgb)).unsqueeze(0)
            with torch.no_grad():
                pred = model(inp)
            # FIX #2 (already correct): sigmoid THEN threshold
            seg_prob = torch.sigmoid(pred).squeeze().cpu().numpy()
            seg_bin  = (seg_prob > seg_thresh).astype(np.uint8)
            seg_mask = cv2.resize(seg_bin, (cw, ch),
                                  interpolation=cv2.INTER_NEAREST)
            seg_rgb  = crop_rgb.copy();  seg_rgb[seg_mask == 0] = 0
            seg_bgr  = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)
            seg_bgr[seg_mask == 0] = 0

        if seg_mask.sum() < 100:
            st.error("❌ LinkNet34 could not segment the face.")
            st.stop()

        with st.expander("📌 Stage 1 — LinkNet34 Segmentation", expanded=True):
            s1a, s1b, s1c = st.columns(3)
            s1a.image(crop_rgb,       caption="Crop",      width=200)
            s1b.image(seg_mask * 255, caption="Seg mask",   width=200, clamp=True)
            s1c.image(seg_rgb,        caption="Segmented",  width=200)

        # ═════════════════════════════════════════════════════
        #  STAGE 2 — YCrCb / HSV skin masking
        # ═════════════════════════════════════════════════════
        with st.spinner(f"Stage 2 — {cspace.upper()} skin mask …"):
            raw_skin = (skin_threshold_ycrcb(seg_bgr)
                        if cspace == "ycrcb"
                        else skin_threshold_hsv(seg_bgr))
            refined   = refine_mask(raw_skin)
            skin_bin  = (refined > 0).astype(np.uint8)
            skin_only = crop_rgb.copy();  skin_only[skin_bin == 0] = 0
            skin_px   = crop_rgb[skin_bin == 1]

        with st.expander(
            f"🔬 Stage 2 — {cspace.upper()} Skin Mask "
            f"({len(skin_px):,} px)", expanded=True
        ):
            s2a, s2b, s2c = st.columns(3)
            s2a.image(seg_rgb,        caption="Segmented",  width=200)
            s2b.image(skin_bin * 255, caption="Skin mask",  width=200, clamp=True)
            s2c.image(skin_only,      caption="Skin only",  width=200)

        if len(skin_px) < 100:
            st.warning("⚠️ Too few skin pixels for classification.")
            st.stop()

        # ═════════════════════════════════════════════════════
        #  PATH 1 — K-Means + Palette
        # ═════════════════════════════════════════════════════
        dom_cols, dom_cnts = dominant_colors_kmeans(skin_px, k=n_k)

        if dom_cols:
            rep_rgb = (
                tuple(np.array(dom_cols[:2], np.float64).mean(axis=0).astype(int))
                if len(dom_cols) >= 2 else dom_cols[0]
            )
            p1_label, p1_pal_hex, p1_your_hex, p1_dist = classify_tone(rep_rgb)
        else:
            p1_label, p1_pal_hex, p1_your_hex, p1_dist = classify_tone(
                skin_px.mean(axis=0)
            )
            dom_cols, dom_cnts = [], []

        # ═════════════════════════════════════════════════════
        #  PATH 2 — Stone Library
        #  FIX #3: pass ORIGINAL crop, not mangled seg_rgb
        #  Stone runs its own face detection internally —
        #  feeding it a black-background image breaks that.
        # ═════════════════════════════════════════════════════
        p2_ok  = False
        p2_hex = p2_lbl = "N/A"
        p2_acc = 0.0
        s_rpt  = {}

        with st.spinner("Path 2 — Stone classifier …"):
            # PRIMARY: give Stone the clean crop (it does its own detection)
            s_res, s_rpt = run_stone(crop_rgb)
            p2_ok = (s_res is not None and bool(s_res.get("faces")))

            if not p2_ok:
                # FALLBACK: try original full image
                s_res, s_rpt = run_stone(orig_np)
                p2_ok = (s_res is not None and bool(s_res.get("faces")))

            if p2_ok:
                fi     = s_res["faces"][0]
                p2_hex = fi.get("skin_tone", "?")
                p2_lbl = fi.get("tone_label", "?")
                p2_acc = fi.get("accuracy", 0)

        # ═════════════════════════════════════════════════════
        #  RESULTS — THREE TABS
        # ═════════════════════════════════════════════════════
        st.divider()
        st.subheader("🔀 Classification Results")

        tab1, tab2, tab3 = st.tabs([
            "⚖️ Compare",
            "🎯 Path 1 · K-Means",
            "🪨 Path 2 · Stone",
        ])

        # ── TAB 1 — Compare ──────────────────────
        with tab1:
            st.markdown("**⚖️ Side-by-Side**")

            left, right = st.columns(2)

            with left:
                st.caption("🎯 K-Means")
                st.markdown(
                    swatch(p1_your_hex, f"{p1_label} · {p1_your_hex}", h=70),
                    unsafe_allow_html=True,
                )
                st.metric("Tone",     p1_label)
                st.metric("Colour",   p1_your_hex)
                st.metric("Palette",  p1_pal_hex)
                st.metric("Distance", f"{p1_dist:.1f}")

            with right:
                st.caption("🪨 Stone")
                if p2_ok:
                    st.markdown(
                        swatch(p2_hex, f"{p2_lbl} · {p2_hex}", h=70),
                        unsafe_allow_html=True,
                    )
                    st.metric("Tone",     p2_lbl)
                    st.metric("HEX",      p2_hex)
                    st.metric("Accuracy", f"{p2_acc}")
                else:
                    st.warning("Stone N/A")

            # Agreement badge
            if p2_ok:
                p1_broad = (
                    "Fair"  if "fair" in p1_label.lower() else
                    "Dark"  if "Dark" in p1_label else "Brown"
                )
                if p1_broad == p2_lbl:
                    st.success(f"✅ Agree: K-Means → {p1_broad} | Stone → {p2_lbl}")
                else:
                    st.warning(f"⚠️ Disagree: K-Means → {p1_broad} | Stone → {p2_lbl}")

            # Pipeline strip — compact
            with st.expander("📊 Full Pipeline Strip", expanded=False):
                p0, p1_, p2_, p3_, p4_ = st.columns(5)
                p0.image(orig_np,         caption="Original",  width=150)
                p1_.image(crop_rgb,       caption="Crop",      width=150)
                p2_.image(seg_rgb,        caption="Segment",   width=150)
                p3_.image(skin_bin * 255, caption="Mask",      width=150, clamp=True)
                p4_.image(skin_only,      caption="Skin",      width=150)

        # ── TAB 2 — K-Means ─────────────────────
        with tab2:
            st.markdown("**🎯 K-Means + Palette Match**")

            m1, m2, m3 = st.columns(3)
            m1.metric("Tone", p1_label)
            m2.metric("Colour", p1_your_hex)
            m3.metric("Pixels", f"{len(skin_px):,}")

            sw1, sw2 = st.columns(2)
            sw1.markdown(swatch(p1_your_hex, f"You — {p1_your_hex}"), unsafe_allow_html=True)
            sw2.markdown(swatch(p1_pal_hex, f"{p1_label} — {p1_pal_hex}"), unsafe_allow_html=True)

            if dom_cols:
                st.caption("Dominant clusters (ranked):")
                kcols = st.columns(len(dom_cols))
                for cw, ((cr, cg, cb), cnt) in zip(kcols, zip(dom_cols, dom_cnts)):
                    hx = f"#{cr:02X}{cg:02X}{cb:02X}"
                    tag = f"★ {hx}" if (cr, cg, cb) == dom_cols[0] else hx
                    cw.markdown(swatch(hx, tag, h=40), unsafe_allow_html=True)
                    cw.caption(f"{cnt:,} px")

            with st.expander("📋 Full 12-Tone Palette"):
                for lbl, (pr, pg, pb) in TONE_PALETTE.items():
                    hx  = f"#{pr:02X}{pg:02X}{pb:02X}"
                    tag = " ← **YOU**" if lbl == p1_label else ""
                    st.markdown(
                        f'<span style="display:inline-block;width:16px;'
                        f'height:16px;background:{hx};border-radius:3px;'
                        f'border:1px solid #444;vertical-align:middle;'
                        f'margin-right:6px"></span>`{lbl}` `{hx}`{tag}',
                        unsafe_allow_html=True,
                    )

        # ── TAB 3 — Stone ────────────────────────
        with tab3:
            st.markdown("**🪨 Stone Skin Tone Classifier**")

            if p2_ok:
                m1, m2, m3 = st.columns(3)
                m1.metric("Tone", p2_lbl)
                m2.metric("HEX", p2_hex)
                m3.metric("Accuracy", f"{p2_acc}")

                st.markdown(
                    swatch(p2_hex, f"Stone — {p2_lbl} — {p2_hex}"),
                    unsafe_allow_html=True,
                )

                if isinstance(s_rpt, dict) and s_rpt:
                    fid = list(s_rpt.keys())[0]
                    rpt_img = s_rpt[fid]
                    if rpt_img is not None:
                        with st.expander("Stone Visual Report"):
                            st.image(rpt_img, width="stretch")

                with st.expander("📋 Stone 11-Colour Palette"):
                    for hx, lb in zip(STONE_PALETTE, STONE_LABELS):
                        tag = (" ← **MATCHED**"
                               if hx.upper() == p2_hex.upper() else "")
                        st.markdown(
                            f'<span style="display:inline-block;width:16px;'
                            f'height:16px;background:{hx};border-radius:3px;'
                            f'border:1px solid #444;vertical-align:middle;'
                            f'margin-right:6px"></span>`{lb}` `{hx}`{tag}',
                            unsafe_allow_html=True,
                        )
            else:
                st.error(
                    f"❌ Stone failed.  "
                    f"{s_rpt if isinstance(s_rpt, str) else 'no faces found'}"
                )

        # ── TAB 3 — Compare ──────────────────────
        with tab3:
            st.markdown("**⚖️ Side-by-Side**")

            left, right = st.columns(2)

            with left:
                st.caption("🎯 K-Means")
                st.markdown(
                    swatch(p1_your_hex, f"{p1_label} · {p1_your_hex}", h=70),
                    unsafe_allow_html=True,
                )
                st.metric("Tone",     p1_label)
                st.metric("Colour",   p1_your_hex)
                st.metric("Palette",  p1_pal_hex)
                st.metric("Distance", f"{p1_dist:.1f}")

            with right:
                st.caption("🪨 Stone")
                if p2_ok:
                    st.markdown(
                        swatch(p2_hex, f"{p2_lbl} · {p2_hex}", h=70),
                        unsafe_allow_html=True,
                    )
                    st.metric("Tone",     p2_lbl)
                    st.metric("HEX",      p2_hex)
                    st.metric("Accuracy", f"{p2_acc}")
                else:
                    st.warning("Stone N/A")

            # Agreement badge
            if p2_ok:
                p1_broad = (
                    "Fair"  if "fair" in p1_label.lower() else
                    "Dark"  if "Dark" in p1_label else "Brown"
                )
                if p1_broad == p2_lbl:
                    st.success(f"✅ Agree: K-Means → {p1_broad} | Stone → {p2_lbl}")
                else:
                    st.warning(f"⚠️ Disagree: K-Means → {p1_broad} | Stone → {p2_lbl}")

            # Pipeline strip — compact
            with st.expander("📊 Full Pipeline Strip", expanded=False):
                p0, p1_, p2_, p3_, p4_ = st.columns(5)
                p0.image(orig_np,         caption="Original",  width="stretch")
                p1_.image(crop_rgb,       caption="Crop",      width="stretch")
                p2_.image(seg_rgb,        caption="Segment",   width="stretch")
                p3_.image(skin_bin * 255, caption="Mask",      width="stretch", clamp=True)
                p4_.image(skin_only,      caption="Skin",      width="stretch")