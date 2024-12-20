from fish_speech.i18n import i18n
import base64
from pathlib import Path

# 读取并编码图标文件
icon_path = Path("fish_speech/webui/icon.ico")
try:
    with open(icon_path, "rb") as f:
        icon_base64 = base64.b64encode(f.read()).decode()
except:
    icon_base64 = ""  # 如果图标文件不存在，使用空字符串

HEADER_MD = f"""<div align="center" style="color: #4ade80;">

# Fish Speech

<div style="display: flex; justify-content: center; align-items: center; margin: 10px 0;">
  <a href="https://space.bilibili.com/259012968" style="margin: 0 2px; display: flex; align-items: center;">
    <img src="data:image/x-icon;base64,{icon_base64}" style="height: 20px; margin-right: 5px;" alt="icon">
    <img src='https://img.shields.io/badge/整合包制：文抑青年-B站主页-ff69b4?style=flat&logo=bilibili' alt='bilibili'>
  </a>
  <a href="https://pd.qq.com/s/6778ns96u" style="margin: 0 2px;">
    <img src='https://img.shields.io/badge/QQ频道-pd85078065-12B7F5?style=flat&logo=tencentqq' alt='qq'>
  </a>
  <a href="https://qm.qq.com/q/l2YU4CQoN2" style="margin: 0 2px;">
    <img src='https://img.shields.io/badge/AI技术交流群-728870602-12B7F5?style=flat&logo=tencentqq' alt='qq'>
  </a>
</div>

</div>

<div style="color: #e0e0e0;">
{i18n("A text-to-speech model based on VQ-GAN and Llama developed by [Fish Audio](https://fish.audio).")}  

{i18n("You can find the source code [here](https://github.com/fishaudio/fish-speech) and models [here](https://huggingface.co/fishaudio/fish-speech-1.5).")}  

{i18n("Related code and weights are released under CC BY-NC-SA 4.0 License.")}  

{i18n("We are not responsible for any misuse of the model, please consider your local laws and regulations before using it.")}  
</div>"""

TEXTBOX_PLACEHOLDER = i18n("Put your text here.")
