[English](README.md) | [简体中文](README_zh_CN.md)

# ComfyUI_FishSpeech_EX
该插件针对Fish-Speech-1.5版本进行适配【仅适用于1.5版本，各版本插件都不一致】：
1. 该插件参考ComfyUI-fish-speech插件进行了优化，修改了整体配置地址与安装方式。
2. 完善了插件所需的Python库，主要是vector-quantize-pytorch，如果未装该库会导致音质不佳。`这个问题困扰了我几天，我把整个FishSpeech插件都找了一篇，查到采样步骤才发现这个问题，如果这个问题同样困扰了你，麻烦点个赞，谢谢！`

#### 具体节点：

- **EX_AudioToPrompt**
1. **audio**: ComfyUI音频。
2. **vqgan**: VQGAN模型。
3. **restored_audio**: 解析后的音频。
4. **prompt_tokens**: 提示音频对应的token。

- **EX_Prompt2Semantic**
1. **prompt_tokens**: 输入提示音频对应的token。
2. **codes**: 生成的音频Code。

- **EX_LoadVQGAN**
加载VQGAN模型，输入模型路径，输出模型。

- **EX_Semantic2Image**
解析音频Code，输出对应的音频。

- **EX_SaveAudioToMp3**
保存音频到MP3文件。

## 工作流
![workflow.png](./workflow/show.png)

## 参考资料
- [AnyaCoder/ComfyUI-fish-speech](https://github.com/AnyaCoder/ComfyUI-fish-speech) - Official Implementaion
- [fishaudio/fish-speech](https://github.com/AnyaCoder/ComfyUI-fish-speech) - SOTA Open Source TTS.

