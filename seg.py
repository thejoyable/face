import streamlit as st
import torch
from torch import nn
from torchvision import models
from torchvision import transforms
import numpy as np
import cv2
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av


# -----------------------------
# Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------
# LinkNet Model
# -----------------------------
def conv3x3(in_, out):
    return nn.Conv2d(in_, out, 3, padding=1)


class ConvRelu(nn.Module):
    def __init__(self, in_, out):
        super().__init__()
        self.conv = conv3x3(in_, out)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.activation(self.conv(x))


class DecoderBlockLinkNet(nn.Module):
    def __init__(self, in_channels, n_filters):
        super().__init__()

        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)

        self.deconv2 = nn.ConvTranspose2d(
            in_channels // 4,
            in_channels // 4,
            kernel_size=4,
            stride=2,
            padding=1
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

    def __init__(self, num_classes=1, pretrained=True):

        super().__init__()

        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=pretrained)

        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
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
        self.finalrelu1 = nn.ReLU(inplace=True)
        self.finalconv2 = nn.Conv2d(32, 32, 3)
        self.finalrelu2 = nn.ReLU(inplace=True)
        self.finalconv3 = nn.Conv2d(32, num_classes, 2, padding=1)

    def forward(self, x):

        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)

        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)

        f1 = self.finaldeconv1(d1)
        f2 = self.finalrelu1(f1)
        f3 = self.finalconv2(f2)
        f4 = self.finalrelu2(f3)
        f5 = self.finalconv3(f4)

        return torch.sigmoid(f5)


# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():

    model = LinkNet34()

    model.load_state_dict(
        torch.load("linknet.pth", map_location=device)
    )

    model.to(device)
    model.eval()

    return model


model = load_model()


# -----------------------------
# Transform (FAST)
# -----------------------------
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256,256)),
    transforms.ToTensor()
])


# -----------------------------
# Video Processor
# -----------------------------
class FaceSegmentation(VideoProcessorBase):

    def recv(self, frame):

        img = frame.to_ndarray(format="bgr24")

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img_tensor = transform(rgb).unsqueeze(0).to(device)

        with torch.no_grad():
            pred = model(img_tensor)

        mask = pred.squeeze().cpu().numpy()

        mask = (mask > 0.5).astype(np.uint8)

        mask = cv2.resize(mask, (img.shape[1], img.shape[0]))

        face = img.copy()
        face[mask == 0] = 0

        return av.VideoFrame.from_ndarray(face, format="bgr24")


# -----------------------------
# Streamlit UI
# -----------------------------
st.title("Real-Time Face Segmentation (LinkNet34)")

st.write("Using webcam with LinkNet segmentation")

webrtc_streamer(
    key="face-segmentation",
    video_processor_factory=FaceSegmentation,
    media_stream_constraints={
        "video": True,
        "audio": False
    }
)