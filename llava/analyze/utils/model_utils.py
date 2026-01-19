import torch
from llava.model.builder import load_pretrained_model as load_llava_model
from llava.conversation import conv_templates
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.mm_utils import tokenizer_image_token
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os

cross_attention_outputs = []

def hook_fn(module, input, output):
    cross_attention_outputs.append(output)

def register_hooks(model):
    for name, module in model.named_modules():
        if 'self_attn' in name.lower():
            module.register_forward_hook(hook_fn)
            break  # 只注册第一个 self attention 层的钩子

def model_inference(engine, model, tokenizer, image, prompt, processor, max_new_tokens, output_dir):
    
    image_tensor =  processor.preprocess([image], return_tensors='pt')['pixel_values'].to(torch.float16).cuda()
    
    if model.config.mm_use_im_start_end:
        inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + prompt
    else:
        inp = DEFAULT_IMAGE_TOKEN + '\n' + prompt
    
    conv_mode = 'llava_v1'
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    
    register_hooks(model)
    
    with torch.inference_mode():
        generated_ids = model.generate(
            input_ids,
            images=image_tensor.unsqueeze(0),
            do_sample=False,
            temperature=1,
            max_new_tokens=max_new_tokens,
            min_new_tokens=1,
            )
    
    predicted_answers = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    # t-SNE analysis
    os.makedirs(output_dir, exist_ok=True)
    for i, output in enumerate(cross_attention_outputs):
        output = output[0].detach().cpu().numpy()
        # 将三维数组展平成二维数组
        _, seq_len, hidden_size = output.shape
        output_flat = output.reshape(seq_len, hidden_size)
        
        # 检查 n_samples 和 n_features 是否大于等于 2
        if output_flat.shape[0] >= 2 and output_flat.shape[1] >= 2:
            tsne = TSNE(n_components=2)
            tsne_result = tsne.fit_transform(output_flat)
            plt.figure()
            plt.scatter(tsne_result[:, 0], tsne_result[:, 1])
            plt.title(f't-SNE of Self Attention Layer {i+1}')
            plt.savefig(os.path.join(output_dir, f'tsne_layer_{i+1}.png'))
            plt.close()
        else:
            print(f"Skipping t-SNE for layer {i+1} due to insufficient samples or features.")
    
    return predicted_answers

def load_model(model_path, args=None):
    tokenizer, model, image_processor, context_len = load_llava_model(model_path=model_path, model_base=None, model_name='llava', 
                                                                      attn_implementation='flash_attention_2', torch_dtype='float16', device_map='cuda',)
    processor = image_processor
    return model, tokenizer, processor