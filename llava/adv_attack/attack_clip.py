from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision.transforms import transforms
import torch
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
import copy
import torch.nn.functional as F
import os
from PIL import Image
import torchvision.transforms as T

class CustomImageDataset(Dataset):
    def __init__(self, folder_path, image_processor, model_config, sample_size=None, seed=42):
        # 获取所有图片文件路径
        all_image_paths = [
            os.path.join(folder_path, filename)
            for filename in os.listdir(folder_path)
            if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp'))
        ]
        
        # 如果指定了样本大小，进行随机抽样
        if sample_size is not None and sample_size < len(all_image_paths):
            import random
            random.seed(seed)  # 设置随机种子以确保可重复性
            self.image_paths = random.sample(all_image_paths, sample_size)
            print(f"Randomly sampled {sample_size} images from {len(all_image_paths)} total images")
        else:
            self.image_paths = all_image_paths
            
        self.image_processor = image_processor
        self.model_config = model_config
        
        # 添加保存原始文件名的属性
        self.filenames = [os.path.basename(path) for path in self.image_paths]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        image_tensor = process_images([image], self.image_processor, self.model_config)[0]
        return image_tensor, self.filenames[idx]

class ComputeLossWrapper:
    def __init__(self, embedding_orig):
        self.embedding_orig = embedding_orig

    def __call__(self, embedding):
        return compute_loss(
            embedding=embedding,
            embedding_orig=self.embedding_orig
        )

def project_perturbation(perturbation, eps, norm):
    if norm in ['inf', 'linf', 'Linf']:
        pert_normalized = torch.clamp(perturbation, -eps, eps)
        return pert_normalized
    elif norm in [2, 2.0, 'l2', 'L2', '2']:
        pert_normalized = torch.renorm(perturbation, p=2, dim=0, maxnorm=eps)
        return pert_normalized
    else:
        raise NotImplementedError(f'Norm {norm} not supported')
    
def normalize_grad(grad, p):
    if p in ['inf', 'linf', 'Linf']:
        return grad.sign()
    elif p in [2, 2.0, 'l2', 'L2', '2']:
        bs = grad.shape[0]
        grad_flat = grad.view(bs, -1)
        grad_normalized = F.normalize(grad_flat, p=2, dim=1)
        return grad_normalized.view_as(grad)
      
def l2(out, targets, reduction='none'):
    assert out.shape == targets.shape
    assert out.shape[0] > 1
    squared_error_batch = F.mse_loss(out, targets, reduction='none')
    if reduction == 'mean':
        squared_error_batch = torch.mean(squared_error_batch.sum(dim=1))
    else:
        squared_error_batch = squared_error_batch.sum(dim=1)
        assert squared_error_batch.shape == (out.shape[0],)
    return squared_error_batch

def compute_loss(embedding, embedding_orig, reduction='mean'):
    loss = l2(out=embedding, targets=embedding_orig, reduction=reduction)
    return loss

def normalize(image_tensor, mean, std):
    return (image_tensor - mean) / std

def denormalize(image_tensor, mean, std):
    return image_tensor * std + mean

def tensor_to_image(tensor):
    # 确保tensor在CPU上并且形状正确
    tensor = tensor.squeeze(0)
    
    # 反归一化到[0,1]范围
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(-1, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(-1, 1, 1)
    tensor = tensor * std + mean
    
    # 将tensor转换为PIL Image
    transform = T.ToPILImage()
    image = transform(tensor)
    return image

def pgd(
    model,
    loss_fn,
    data_clean,
    norm,
    eps,
    iterations,
    stepsize,
    output_normalize=False,
    perturbation=None,
    mode='max',
    momentum=0.9,
    verbose=True,
):
    batch_size = data_clean.shape[0]

    image_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1).expand(batch_size, -1, -1, -1)
    image_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).expand(batch_size, -1, -1, -1)

    eps_tensor = torch.full((batch_size, 3, 1, 1), eps / 255)
    dis_tensor = torch.full((batch_size, 3, 1, 1), 0)
    
    eps = normalize(eps_tensor, image_mean, image_std)
    dis = normalize(dis_tensor, image_mean, image_std)
    eps = eps - dis

    stepsize_tensor = torch.full((batch_size, 3, 1, 1), stepsize / 255)
    stepsize = normalize(stepsize_tensor, image_mean, image_std)
    stepsize = stepsize - dis

    perturbation = normalize(torch.zeros_like(data_clean).uniform_(-4 / 255, 4 / 255), image_mean, image_std)  - dis
    perturbation.requires_grad_(True)

    velocity = torch.zeros_like(data_clean)

    for i in range(iterations):
        perturbation.requires_grad = True
        with torch.enable_grad():
            image_features = model.get_model().get_vision_tower()(data_clean + perturbation) 
            out = image_features  
            loss = loss_fn(out)
            if verbose:
                print(f'[{i}] {loss.item():.5f}')

        with torch.no_grad():
            gradient = torch.autograd.grad(loss, perturbation)[0]

            if gradient.isnan().any():
                print(f'attention: nan in gradient ({gradient.isnan().sum()})')
                gradient[gradient.isnan()] = 0.

            gradient = normalize_grad(gradient, p=norm)
            velocity = momentum * velocity + gradient
            velocity = normalize_grad(velocity, p=norm)

            if mode == 'min':
                perturbation = perturbation - stepsize * velocity
            elif mode == 'max':
                perturbation = perturbation + stepsize * velocity
            else:
                raise ValueError(f'Unknown mode: {mode}')
            
            #perturbation = perturbation
            # print("for perturbation before projection")
            # print(perturbation.max())
            # print(perturbation.min())
            perturbation = project_perturbation(perturbation, eps, norm)
            # print("for perturbation after projection")
            # print(perturbation.max())
            # print(perturbation.min())
            denormalized_image = denormalize(data_clean + perturbation, image_mean, image_std)
            # print("for data_clean + perturbation")
            # print((data_clean + perturbation).max())
            # print((data_clean + perturbation).min())
            #denormalized_image = denormalize(data_clean, image_mean, image_std)

            # print(denormalized_image.max())
            # print(denormalized_image.min())
            denormalized_image = torch.clamp(denormalized_image, 0, 1)
            perturbation = normalize(denormalized_image, image_mean, image_std) - data_clean
            # print("for perturbation after normalize")
            # print(perturbation.max())
            # print(perturbation.min())
    return data_clean + perturbation.detach()

if __name__ == "__main__":
    # 设置设备和路径
    device = "cuda"
    torch.set_default_device(device)
    
    # 设置模型路径和输出路径
    model_path = "./llava-v1.5-7b"  # 请修改为你的模型路径
    output_dir = "./adv_examples/mm-vet/ori"  # 对抗样本保存路径
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 加载模型
    print("Loading model...")
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, None, model_name, device=device, torchdtype=torch.float32
    )
    
    # for p in model.get_model().mm_projector.parameters():
    #     p.requires_grad = True
    # 准备数据
    image_folder = "./playground/data/eval/mm-vet/mm-vet/images"  
    dataset = CustomImageDataset(
        folder_path=image_folder,
        image_processor=image_processor,
        model_config=model.config,
        sample_size=None,  # 随机抽取5000张图片
        seed=42  # 设置随机种子以确保可重复性
    )
    data_loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)
    
    # 获取vision tower
    vision_tower = model.get_model().get_vision_tower()
    #projector = model.get_model().mm_projector
    
    print("Generating adversarial examples...")
    
    from tqdm import tqdm

    for batch_idx, (data, filenames) in enumerate(tqdm(data_loader, desc="Processing batches")):
        print(f"Processing batch {batch_idx + 1}...")
        data = data.to(device)
        
        # 获取原始特征
        with torch.no_grad():
            feature_orig = vision_tower(data)
            #embedding_orig = projector(feature_orig)
        
        # 生成对抗样本
        loss_inner_wrapper = ComputeLossWrapper(feature_orig)

        model.eval()
        
        # data_adv = pgd(
        #     model=model,
        #     loss_fn=loss_inner_wrapper,
        #     data_clean=data,
        #     norm='linf',
        #     eps=4,  # 扰动大小
        #     iterations=10, 
        #     stepsize=1,
        #     output_normalize=False,
        #     mode='max',
        #     verbose=True
        # )
        data_adv = data
        # 保存对抗样本
        for idx, (adv_sample, filename) in enumerate(zip(data_adv, filenames)):
            adv_image = tensor_to_image(adv_sample)
            save_path = os.path.join(output_dir, f"{filename}")
            adv_image.save(save_path)
            print(f"Saved adversarial example to {save_path}")
        
        print(f"Batch {batch_idx + 1} completed")
    
    print("Adversarial example generation completed!")