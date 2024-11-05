import torch
from nerf.utils import *

class InferenceRunner:
    def __init__(self, trainer, checkpoint_path=None, device=None):
        """
        初始化推理过程
        :param trainer: 已经定义好的 Trainer 对象
        :param checkpoint_path: 保存的模型权重路径
        :param device: 推理设备 (CPU or GPU)
        """
        self.trainer = trainer
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.trainer.device = self.device

        # 加载训练好的模型权重
        if checkpoint_path:
            print(f"Loading checkpoint from {checkpoint_path}")
            self.trainer.load_checkpoint(checkpoint=checkpoint_path, model_only=True)
        else:
            print("Warning: No checkpoint provided, using randomly initialized model.")

    def infer(self, data_loader):
        """
        推理过程：对给定的数据执行推理，输出预测结果
        :param data_loader: 包含推理数据的数据加载器
        :return: 推理结果
        """
        self.trainer.model.eval()  # 设置模型为评估模式

        all_preds = []
        all_truths = []

        with torch.no_grad():  # 在推理过程中不计算梯度，节省内存
            for data in data_loader:
                # 将数据加载到设备上
                data = data.to(self.device)

                # 通过模型进行前向推理
                with torch.cuda.amp.autocast(enabled=self.trainer.fp16):
                    preds, truths, loss = self.trainer.train_step(None, data, None)  # train_step 执行前向传播

                all_preds.append(preds)
                all_truths.append(truths)

        return all_preds, all_truths

# 使用示例
if __name__ == "__main__":
    # 假设 Trainer 已经定义好，并且你已经初始化了它
    trainer = Trainer(name="experiment", opt={}, model=None, device=torch.device("cuda"))

    # 推理器实例化
    inference_runner = InferenceRunner(trainer, checkpoint_path="path/to/checkpoint.pth")

    # 假设已经有一个数据加载器 `test_loader`，其中包含需要进行推理的数据
    test_loader = ...  # 你的数据加载器

    # 执行推理过程
    predictions, ground_truths = inference_runner.infer(test_loader)

    # 输出推理结果
    print("Inference completed. Predictions:", predictions)
