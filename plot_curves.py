import matplotlib.pyplot as plt
import matplotlib

# ========== 解决中文显示问题 ==========
# 方法1：尝试使用系统中文字体（推荐）
try:
    # Windows 常用中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Zen Hei', 'Noto Sans CJK SC']
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
except:
    # 如果上述不行，就用英文标签（避免警告）
    use_english = True
else:
    use_english = False

# ========== 训练数据（根据你的日志录入） ==========
epochs = list(range(1, 21))
train_loss = [1.157, 0.873, 0.792, 0.766, 0.753, 0.665, 0.707, 0.722, 0.714, 0.656,
              0.678, 0.684, 0.649, 0.664, 0.687, 0.682, 0.627, 0.653, 0.655, 0.664]
train_acc = [0.571, 0.689, 0.712, 0.716, 0.723, 0.754, 0.736, 0.746, 0.742, 0.768,
             0.756, 0.752, 0.762, 0.755, 0.755, 0.761, 0.781, 0.756, 0.754, 0.761]
val_acc = [0.664, 0.670, 0.686, 0.678, 0.670, 0.719, 0.662, 0.674, 0.709, 0.660,
           0.674, 0.658, 0.688, 0.700, 0.692, 0.711, 0.721, 0.700, 0.698, 0.725]

# ========== 绘图 ==========
plt.figure(figsize=(12, 4))

# 子图1：训练损失曲线
plt.subplot(1, 2, 1)
plt.plot(epochs, train_loss, 'b-o', linewidth=2, markersize=4)
plt.xlabel('Epoch' if use_english else '训练轮次')
plt.ylabel('Loss' if use_english else '损失值')
plt.title('Training Loss' if use_english else '训练损失变化')
plt.grid(True, linestyle='--', alpha=0.6)

# 子图2：准确率曲线
plt.subplot(1, 2, 2)
plt.plot(epochs, train_acc, 'g-o', linewidth=2, markersize=4, label='Train Acc' if use_english else '训练准确率')
plt.plot(epochs, val_acc, 'r-s', linewidth=2, markersize=4, label='Val Acc' if use_english else '验证准确率')
plt.xlabel('Epoch' if use_english else '训练轮次')
plt.ylabel('Accuracy' if use_english else '准确率')
plt.title('Accuracy Curves' if use_english else '训练与验证准确率曲线')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig('training_curves.png', dpi=300)
plt.show()

print("曲线图已保存为 training_curves.png")