import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import torch.nn.functional as F
import torchvision.transforms as T

class JailbreakPrompts:
    def __init__(self, csv_path):
        # 读取CSV文件
        self.df = pd.read_csv(csv_path)
        self.prompts = self.df['prompt'].tolist()
        self.targets = self.df['target'].tolist()
    
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        return self.prompts[idx], self.targets[idx]

class ComputeLossWrapper:
    def __init__(self, model, tokenizer, prompt, target):
        self.model = model
        self.tokenizer = tokenizer
        self.prompt = prompt
        self.target = target
    
    def __call__(self, image_features):
        # 构建输入
        conv = conv_templates[model_name].copy()
        conv.append_message(conv.roles[0], self.prompt)
        prompt = conv.get_prompt()
        
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').to(image_features.device)
        
        # 计算模型输出
        outputs = self.model(
            input_ids=input_ids,
            images=image_features,
            return_dict=True
        )
        
        # 使用目标回复计算损失
        target_ids = self.tokenizer(self.target, return_tensors='pt').input_ids.to(image_features.device)
        logits = outputs.logits
        
        # 计算交叉熵损失 - 最大化目标回复的概率
        loss = -F.cross_entropy(logits[:, -target_ids.size(1):, :], target_ids)
        return loss

def main():
    # 设置设备和路径
    device = "cuda"
    torch.set_default_device(device)
    
    # 加载模型和数据集
    model_path = "/path/to/model"
    output_base_dir = "./adv_examples"
    jailbreak_csv = "path_to_jailbreak.csv"
    
    # 加载模型
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, None, model_name, device=device, torchdtype=torch.float32
    )
    
    # 加载数据集
    dataset = CustomImageDataset(
        folder_path="./images",
        image_processor=image_processor,
        model_config=model.config,
    )
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)  # 批大小设为1以便处理
    
    # 加载越狱提示
    jailbreak_data = JailbreakPrompts(jailbreak_csv)
    
    # 对每个图像进行处理
    for batch_idx, (data, filenames) in enumerate(tqdm(data_loader)):
        data = data.to(device)
        current_filename = filenames[0]  # 因为batch_size=1
        
        # 为每个提示创建单独的输出目录
        prompt_output_dir = os.path.join(output_base_dir, f"image_{current_filename}")
        os.makedirs(prompt_output_dir, exist_ok=True)
        
        # 对每个提示进行PGD攻击
        for prompt_idx in range(len(jailbreak_data)):
            prompt, target = jailbreak_data[prompt_idx]
            
            # 创建损失函数
            loss_fn = ComputeLossWrapper(model, tokenizer, prompt, target)
            
            # 进行PGD攻击
            data_adv = pgd(
                model=model,
                loss_fn=loss_fn,
                data_clean=data,
                norm='linf',
                eps=8,
                iterations=10,
                stepsize=2,
                mode='max',
                verbose=True
            )
            
            # 保存对抗样本
            adv_image = tensor_to_image(data_adv[0])  # 取第一个样本因为batch_size=1
            save_path = os.path.join(prompt_output_dir, f"prompt_{prompt_idx}.png")
            adv_image.save(save_path)
            
            # 保存提示信息
            with open(os.path.join(prompt_output_dir, f"prompt_{prompt_idx}_info.txt"), 'w') as f:
                f.write(f"Prompt: {prompt}\nTarget: {target}")
            
            print(f"Processed prompt {prompt_idx + 1}/{len(jailbreak_data)} for image {current_filename}")

if __name__ == "__main__":
    main()