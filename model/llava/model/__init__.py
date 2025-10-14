from .language_model.llava_llama import LlavaConfig, LlavaLlamaForCausalLM
# from .language_model.llava_mpt import LlavaMPTConfig, LlavaMPTForCausalLM
try:
    from .language_model.llava_mpt import LlavaMPTConfig, LlavaMPTForCausalLM
except Exception:
    pass
