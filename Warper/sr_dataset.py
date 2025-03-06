import os
import torch
from torchvision import transforms
from PIL import Image
from datasets import Dataset

# 定义自定义数据集类
class SuperResolutionDataset(Dataset):
    def __init__(self, blurry_dir, high_res_dir, emotion_embedding_path, transform=None):
        self.blurry_dir = blurry_dir
        self.high_res_dir = high_res_dir
        self.emotion_embeddings = torch.load(emotion_embedding_path)
        self.transform = transform

        # 获取所有图片文件名
        self.blurry_images = sorted(os.listdir(blurry_dir))
        self.high_res_images = sorted(os.listdir(high_res_dir))

    def __len__(self):
        return len(self.blurry_images)

    def __getitem__(self, idx):
        # 加载模糊图片
        blurry_image_path = os.path.join(self.blurry_dir, self.blurry_images[idx])
        blurry_image = Image.open(blurry_image_path).convert("RGB")

        # 加载高清图片
        high_res_image_path = os.path.join(self.high_res_dir, self.high_res_images[idx])
        high_res_image = Image.open(high_res_image_path).convert("RGB")

        # 加载表情编码
        emotion_embedding = self.emotion_embeddings[idx]

        # 数据预处理
        if self.transform:
            blurry_image = self.transform(blurry_image)
            high_res_image = self.transform(high_res_image)

        return {
            "blurry_image": blurry_image,
            "high_res_image": high_res_image,
            "emotion_embedding": emotion_embedding,
        }

# 定义数据预处理
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # 调整图片大小
    transforms.ToTensor(),          # 转换为张量
])

# 定义数据集路径
blurry_dir = "your_dataset/blurry_images"
high_res_dir = "your_dataset/high_res_images"
emotion_embedding_path = "your_dataset/emotion_embeddings.pt"

# 加载数据集
dataset = SuperResolutionDataset(blurry_dir, high_res_dir, emotion_embedding_path, transform=transform)

# 将自定义数据集转换为 Hugging Face 的 Dataset 格式
hf_dataset = Dataset.from_dict({
    "blurry_image": [item["blurry_image"] for item in dataset],
    "high_res_image": [item["high_res_image"] for item in dataset],
    "emotion_embedding": [item["emotion_embedding"] for item in dataset],
})