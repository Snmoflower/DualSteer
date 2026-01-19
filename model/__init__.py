def build_model(args):
    if "llava" in args.model_path.lower():
        from .LLaVAModel import LLaVA
        model = LLaVA(args.model_path)
    # elif args.model_name == "LLaMA_Adapter":
    #     from .LLaMA_Adapter import LLaMA_Adapter
    #     model = LLaMA_Adapter(args)
    elif "instructblip" in args.model_path.lower():
        from .InstructBLIP import InstructBLIP
        model = InstructBLIP(args)
    # elif args.model_name == "GPT4V":
    #     from .GPT4V import GPTClient
    #     model = GPTClient()
    elif "minigpt4" in args.model_path.lower():
        from .MiniGPT4 import MiniGPT4
        model = MiniGPT4(args)
    elif "ayavision" in args.model_path.lower():
        from .AyaVision import AyaVision
        model = AyaVision(args)
    elif "qwen3" in args.model_path.lower():
        from .Qwen3 import Qwen3
        model = Qwen3(args.model_path)
    else:
        from .Qwen25 import Qwen25
        model = Qwen25(args)
        
    return model