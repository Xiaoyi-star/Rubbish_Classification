import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset

# ── 1. 数据预处理 ──────────────────────────────────────────
# 训练集：随机翻转 + 随机裁剪，增加数据多样性，防止过拟合
train_transform = transforms.Compose([
    transforms.Resize((128, 128)),        # 缩小尺寸，加快 CPU 训练
    transforms.RandomHorizontalFlip(),    # 随机水平翻转
    transforms.RandomRotation(15),        # 随机旋转 ±15 度
    transforms.ToTensor(),                # 转为张量 (0~1)
    transforms.Normalize(                 # 标准化（ImageNet 均值/标准差）
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
])

# 验证集：只做基础处理，不做随机增强
val_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
])

# ── 2. 加载数据集 ──────────────────────────────────────────
# 用两个独立的 dataset 对象分别应用不同的预处理
# 注意：不能对 random_split 的结果直接设 .dataset.transform，
#       因为两个 subset 共享同一个底层对象，后设的会覆盖先设的
train_data = datasets.ImageFolder('Garbage classification',
                                   transform=train_transform)
val_data   = datasets.ImageFolder('Garbage classification',
                                   transform=val_transform)
class_names = train_data.classes
print("类别：", class_names)

# 按相同的随机种子划分索引，保证两个 dataset 划分一致
n_total = len(train_data)
n_train = int(n_total * 0.8)
n_val   = n_total - n_train
indices = torch.randperm(n_total).tolist()
train_indices = indices[:n_train]
val_indices   = indices[n_train:]

from torch.utils.data import Subset
train_set = Subset(train_data, train_indices)
val_set   = Subset(val_data,   val_indices)

train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
val_loader   = DataLoader(val_set,   batch_size=16, shuffle=False)
print(f"训练集：{n_train} 张，验证集：{n_val} 张")

# ── 3. 加载预训练模型并替换分类头 ─────────────────────────
# MobileNetV2 是轻量级网络，适合 CPU 运行
model = models.mobilenet_v2(weights='IMAGENET1K_V1')

# 冻结主干网络，只训练最后的分类头（加快训练速度）
for param in model.features.parameters():
    param.requires_grad = False

# 将原来的 1000 类分类头替换为 6 类
model.classifier[1] = nn.Linear(model.last_channel, len(class_names))
print("模型加载完成，开始训练...")

# ── 4. 训练配置 ────────────────────────────────────────────
device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')           # 无 GPU 使用 CPU
model     = model.to(device)
criterion = nn.CrossEntropyLoss()         # 多分类交叉熵损失
optimizer = torch.optim.Adam(
    model.classifier.parameters(), lr=0.001)

# ── 5. 训练循环 ────────────────────────────────────────────
EPOCHS    = 20
best_acc  = 0.0

for epoch in range(EPOCHS):
    # 训练阶段
    model.train()
    total_loss, correct = 0, 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct    += (outputs.argmax(1) == labels).sum().item()

    train_acc = correct / n_train

    # 验证阶段
    model.eval()
    val_correct = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            val_correct += (outputs.argmax(1) == labels).sum().item()

    val_acc = val_correct / n_val
    print(f"Epoch [{epoch+1}/{EPOCHS}]  "
          f"Loss: {total_loss/len(train_loader):.3f}  "
          f"Train Acc: {train_acc:.3f}  "
          f"Val Acc: {val_acc:.3f}")

    # 保存验证集最优权重
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), 'best_model.pth')
        print(f"  → 已保存最优权重（Val Acc:{best_acc:.3f}）")

print(f"\n训练完成！最优验证准确率：{best_acc:.3f}")
print("权重已保存至best_model.pth")