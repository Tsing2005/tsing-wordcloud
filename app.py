import streamlit as st
import jieba
from wordcloud import WordCloud
from PIL import Image, ImageOps
import numpy as np
from collections import Counter
import io
import os
import platform
import requests
import re

st.set_page_config(page_title="全格式词云生成器", layout="wide")
st.title("☁️ wordcloud")

# ================= 1. 环境准备函数 =================
def get_default_font():
    if platform.system() == "Windows":
        paths = ["C:/Windows/Fonts/simhei.ttf", "C:/Windows/Fonts/msyh.ttc"]
    elif platform.system() == "Darwin": 
        paths = ["/Library/Fonts/Arial Unicode.ttf", "/System/Library/Fonts/STHeiti Light.ttc"]
    else: 
        paths = ["/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"]
    for p in paths:
        if os.path.exists(p): return p
    return "simhei.ttf"

def process_mask_to_array(image, threshold=240):
    if image.mode in ('RGBA', 'LA') or (image.mode == 'P' and 'transparency' in image.info):
        image = image.convert("RGBA")
        r, g, b, a = image.split()
        if a.getextrema()[0] < 255: 
            mask = np.where(np.array(a) > 10, 0, 255).astype(np.uint8)
            return np.stack([mask]*3, axis=-1)

    img_array = np.array(image.convert("RGB"))
    min_channels = np.min(img_array, axis=2)
    mask = np.where(min_channels >= threshold, 255, 0).astype(np.uint8)
    
    return np.stack([mask]*3, axis=-1)

# ===== 火山引擎 豆包 API 基础配置 =====
DOUBAO_API_KEY = "ark-45dee623-dd63-4749-b67e-b327ba49f8f1-64107"

# 文本模型调用（用于语义匹配）保持不变
def call_ai_api(prompt):
    url = "https://ark.cn-beijing.volces.com/api/v3/responses"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {DOUBAO_API_KEY}"
    }
    body = {
        "model": "doubao-seed-2-0-pro-260215",
        "input": [
            {
                "role": "user",
                "content": [{"type": "input_text", "text": prompt}]
            }
        ]
    }
    res = requests.post(url, json=body, headers=headers, timeout=120)
    if res.status_code != 200:
        raise ValueError(f"文本API请求失败：{res.status_code}，{res.text}")
    data = res.json()
    try:
        if "output" in data and "text" in data["output"]: return data["output"]["text"]
        elif "choices" in data: return data["choices"][0]["message"]["content"]
        else: return str(data)
    except Exception:
        raise ValueError(f"API 返回格式异常：{data}")

def ai_match_mask_filename(text, candidate_files):
    if not candidate_files: raise ValueError("masks 文件夹为空。")
    prompt = f"""你是一个严格遵守规则的助手。请从给定列表中挑一个与文本最相近的文件名。
文本：{text}
候选列表：{candidate_files}
规则：只返回一个文件名，不要加引号或任何说明。"""
    result = call_ai_api(prompt).strip().strip('"').strip("'")
    if result in candidate_files: return result
    for f in candidate_files:
        if f == result or f in result or result in f: return f
    raise ValueError(f"AI 未返回有效文件名：{result}")


# ===== 【核心修改区】AI 画图接口替换为 Seedream =====
def generate_mask_image_by_ai(text):
    # 针对顶刊要求和多主体（如脑肠轴）进行指令调优
    drawing_prompt = f"""请根据以下文本内容提取最能概括其核心概念的具体物品或形状进行作画。
文本内容：{text[:1000]}
生成要求（必须严格遵守）：
1. 本图像将作为国际顶刊（如Nature/Cell）专业学术论文的词云掩码（Mask），必须极度专业、严谨、大方、高级。
2. 必须是高度抽象概括的极简实心纯黑剪影（Silhouette）。无内部细节、无灰度过渡、无边框、无阴影、无文字，背景必须完全纯白，主体必须完全纯黑。
3. 允许根据文本概念生成多个主体（例如“脑肠轴”可以同时出现大脑和肠道），但它们必须在视觉上构成一个协调、连贯、紧凑的整体轮廓。坚决杜绝生硬拼接、散乱分布、像劣质插画一样的排版，绝对不要使用细劣的线条、箭头或关系连接符。
4. 整体外轮廓必须平滑流畅，作为词云背景形状时能保持视觉上的饱满感与图形美感。
"""
    
    url = "https://ark.cn-beijing.volces.com/api/v3/images/generations"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {DOUBAO_API_KEY}"
    }
    
    # 完全采用提供的 Seedream 模型格式
    body = {
        "model": "doubao-seedream-5-0-260128",
        "prompt": drawing_prompt,
        "response_format": "url"
    }

    res = requests.post(url, json=body, headers=headers, timeout=120)
    
    if res.status_code != 200:
        raise ValueError(f"AI生图失败，HTTP状态码：{res.status_code}，响应内容：{res.text}")
        
    data = res.json()
    try:
        # 解析标准生图格式的数据
        image_url = data["data"][0]["url"]
    except Exception:
        raise ValueError(f"解析生图URL失败。API返回：{data}")

    # 获取真实生成的图片数据并读取
    img_res = requests.get(image_url, timeout=120)
    img_res.raise_for_status()
    return Image.open(io.BytesIO(img_res.content))
# ========================================================


# ================= 2. 侧边栏：参数与模版选择 =================
st.sidebar.header("🎨 设计面板")

fonts_dir = "fonts"
if os.path.exists(fonts_dir):
    font_files = [f for f in os.listdir(fonts_dir) if f.lower().endswith(('.ttf', '.otf', '.ttc'))]
else:
    font_files = []

if font_files:
    selected_font = st.sidebar.selectbox("选择字体：", font_files)
    font_p = os.path.join(fonts_dir, selected_font)
else:
    st.sidebar.warning("⚠️ 未在 fonts 文件夹中检测到字体")
    font_p = st.sidebar.text_input("字体路径", value=get_default_font())

st.sidebar.subheader("色彩方案")
bg_color = st.sidebar.color_picker("1. 画布外部背景", "#FFFFFF")
shape_inner_color = st.sidebar.color_picker("2. 形状内部底色", "#FDF5E6")

color_options = ["viridis", "Set1", "plasma", "Dark2", "spring", "magma", "coolwarm", "rainbow"]
text_colormap = st.sidebar.selectbox("3. 文字颜色主题", color_options)

colormap_gradients = {
    "viridis": "linear-gradient(to right, #440154, #31688e, #35b779, #fde725)",
    "Set1": "linear-gradient(to right, #e41a1c, #377eb8, #4daf4a, #984ea3, #ff7f00)",
    "plasma": "linear-gradient(to right, #0d0887, #9c179e, #ed7953, #f0f921)",
    "Dark2": "linear-gradient(to right, #1b9e77, #d95f02, #7570b3, #e7298a, #66a61e)",
    "spring": "linear-gradient(to right, #ff00ff, #ffff00)",
    "magma": "linear-gradient(to right, #000004, #51127c, #b73779, #fc8961, #fcfdbf)",
    "coolwarm": "linear-gradient(to right, #3b4cc0, #dddddd, #b40426)",
    "rainbow": "linear-gradient(to right, #ff0000, #ff7f00, #ffff00, #00ff00, #0000ff, #4b0082, #8b00ff)"
}

st.sidebar.markdown(
    f'<div style="width: 100%; height: 12px; border-radius: 4px; background: {colormap_gradients[text_colormap]}; margin-top: -12px; margin-bottom: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.2);"></div>',
    unsafe_allow_html=True
)

st.sidebar.subheader("📏 字体大小控制")
min_f_size = st.sidebar.slider("最小字体大小", min_value=4, max_value=100, value=12)
max_f_size = st.sidebar.slider("最大字体大小", min_value=50, max_value=500, value=200)

st.sidebar.subheader("图形模版识别")
mask_threshold = st.sidebar.slider("抠图白底容差 (若形状中间有空洞，请调大此值)", min_value=150, max_value=255, value=180, step=5)

# 合并选项为一个单选框
mask_mode = st.sidebar.radio(
    "掩码模版获取方式：",
    ["手动选择本地模版", "上传自定义照片", "AI匹配本地模版", "AI生成新掩码图"]
)

mask_image_to_use = None
supported_formats = ('.png', '.jpg', '.jpeg', '.webp')

# 更新了侧边栏的逻辑判断以适配合并后的选项
if mask_mode == "手动选择本地模版":
    mask_dir = "masks"
    if os.path.exists(mask_dir):
        files = [f for f in os.listdir(mask_dir) if f.lower().endswith(supported_formats)]
        if files:
            selected_file = st.sidebar.selectbox("请选择一个默认图形：", files)
            mask_image_to_use = Image.open(os.path.join(mask_dir, selected_file))
            st.sidebar.image(mask_image_to_use, caption="当前图形预览", width=120)
        else:
            st.sidebar.warning(f"目录 '{mask_dir}' 下没有找到图片。")
elif mask_mode == "上传自定义照片":
    uploaded = st.sidebar.file_uploader("点击上传图片 (支持 WebP, PNG, JPG)", type=["png", "jpg", "jpeg", "webp"])
    if uploaded:
        mask_image_to_use = Image.open(uploaded)
        st.sidebar.image(mask_image_to_use, caption="上传成功", width=120)

# ================= 3. 主界面逻辑 =================
text_area = st.text_area("输入文字内容：", height=150, value="Ageing is accompanied by declining memory function, with extremely heterogeneous manifestation in the human population1. Brain-extrinsic factors influencing cognitive decline, such as gastrointestinal signals, have emerged as attractive targets for peripheral interventions2,3,4,5,6, but the underlying mechanisms remain largely unclear. Here, by charting a high-resolution map of microbiome ageing and its functional consequences throughout the lifespan of mice, we identify a mechanism by which inhibition of gut–brain signalling during ageing results in impaired neuronal activation in the hippocampus and loss of memory encoding. Specifically, accumulation of gut bacteria that produce medium-chain fatty acids, such as Parabacteroides goldsteinii, can drive peripheral myeloid cell inflammation through GPR84 signalling. As a result, the function of vagal afferent neurons is impaired, the interoceptive signal received by the brain is weakened and hippocampal function declines. We leverage this pathway to define interventions that enhance memory in aged mice, such as phage targeting of Parabacteroides, GPR84 inhibition and restoration of vagal activity. These findings indicate a key role for interoceptive dysfunction in brain ageing and suggest that interoceptomimetics that stimulate gut–brain communication may counteract age-associated cognitive decline.")

if text_area:
    seg_list = jieba.lcut(text_area)
    counts = dict(Counter([w for w in seg_list if len(w) > 1 and w.strip()]).most_common())
    
    if not counts:
        st.info("等待输入文本...")
        st.stop()

    max_words = st.slider("选择展示词语的数量：", min_value=1, max_value=len(counts), value=min(50, len(counts)))
    
    words_to_show = st.multiselect(
        "筛选词汇：", 
        options=list(counts.keys()), 
        default=list(counts.keys())[:max_words],
        format_func=lambda w: f"{w} ({counts[w]})"
    )

    # 提前为 AI 掩码及修改面板占一个位，确保它能在按钮的上方瞬间呈现
    mask_display_container = st.container()

    if st.button("🚀 生成高清词云"):
        if not words_to_show:
            st.error("词汇列表为空")
        else:
            try:
                with st.spinner("正在执行高精度抠图与图层合成..."):
                    filtered_data = {w: counts[w] for w in words_to_show}

                    # ===== 使用文本 API 匹配 masks =====
                    if mask_mode == "AI匹配本地模版":
                        mask_dir = "masks"
                        if not os.path.exists(mask_dir):
                            st.error("未找到 'masks' 文件夹。")
                            st.stop()

                        files = [f for f in os.listdir(mask_dir) if f.lower().endswith(supported_formats)]
                        if not files:
                            st.error(f"目录 '{mask_dir}' 下没有找到图片。")
                            st.stop()

                        if st.session_state.get("ai_cache_text") == text_area and st.session_state.get("ai_cache_mode") == mask_mode and st.session_state.get("ai_cache_image") is not None:
                            mask_image_to_use = st.session_state["ai_cache_image"]
                            matched_file = st.session_state["ai_cache_filename"]
                        else:
                            matched_file = ai_match_mask_filename(text_area, files)
                            mask_image_to_use = Image.open(os.path.join(mask_dir, matched_file))
                            st.session_state["ai_cache_text"] = text_area
                            st.session_state["ai_cache_mode"] = mask_mode
                            st.session_state["ai_cache_image"] = mask_image_to_use
                            st.session_state["ai_cache_filename"] = matched_file

                    # ===== 使用画图 API 动态生成新掩码 =====
                    elif mask_mode == "AI生成新掩码图":
                        if st.session_state.get("ai_cache_text") == text_area and st.session_state.get("ai_cache_mode") == mask_mode and st.session_state.get("ai_cache_image") is not None:
                            mask_image_to_use = st.session_state["ai_cache_image"]
                        else:
                            mask_image_to_use = generate_mask_image_by_ai(text_area)
                            st.session_state["ai_cache_text"] = text_area
                            st.session_state["ai_cache_mode"] = mask_mode
                            st.session_state["ai_cache_image"] = mask_image_to_use

                    mask_array = None
                    if mask_image_to_use:
                        mask_array = process_mask_to_array(mask_image_to_use, threshold=mask_threshold)
                    
                    wc = WordCloud(
                        font_path=font_p,
                        background_color=None,
                        mode="RGBA",
                        mask=mask_array,
                        colormap=text_colormap,
                        width=1000 if mask_array is None else mask_array.shape[1],
                        height=1000 if mask_array is None else mask_array.shape[0],
                        contour_width=0,
                        relative_scaling=0.15,
                        min_font_size=min_f_size,
                        max_font_size=max_f_size
                    )
                    wc.generate_from_frequencies(filtered_data)
                    wc_layer = wc.to_image()
                    w, h = wc_layer.size

                    final_canvas = Image.new("RGBA", (w, h), bg_color)
                    
                    if mask_array is not None:
                        inner_layer = Image.new("RGBA", (w, h), shape_inner_color)
                        shape_mask_data = 255 - mask_array[:, :, 0]
                        shape_mask_img = Image.fromarray(shape_mask_data, mode="L")
                        final_canvas = Image.composite(inner_layer, final_canvas, shape_mask_img)

                    final_result = Image.alpha_composite(final_canvas, wc_layer)

                    st.image(final_result, caption="渲染结果", use_container_width=True)

                    st.markdown("---")
                    st.subheader("📥 下载中心")
                    c1, c2 = st.columns(2)
                    
                    png_buf = io.BytesIO()
                    final_result.save(png_buf, format="PNG")
                    c1.download_button("🖼️ 下载 PNG", data=png_buf.getvalue(), file_name="cloud_star.png")
                    
                    pdf_buf = io.BytesIO()
                    final_result.convert("RGB").save(pdf_buf, format="PDF", resolution=300)
                    c2.download_button("📩 下载 PDF", data=pdf_buf.getvalue(), file_name="cloud_star.pdf")

            except Exception as e:
                st.error(f"渲染失败: {e}")

    # 将占位符填充满，如果在上面的按钮逻辑中获取到了新图，它会立刻无缝在这里展示
    with mask_display_container:
        if mask_mode == "AI生成新掩码图" and st.session_state.get("ai_cache_text") == text_area and st.session_state.get("ai_cache_mode") == mask_mode and st.session_state.get("ai_cache_image") is not None:
            st.success("当前缓存的 AI 掩码图如下（随意调整上方参数，该图不会消失）：")
            col1, col2 = st.columns([1, 2])
            with col1:
                st.image(st.session_state["ai_cache_image"], width=200)
            with col2:
                st.markdown("##### ✏️ 对生成的掩码图不满意？")
                modify_prompt = st.text_input("您可以输入修改指令（例如：换成大脑的形状、只保留外轮廓、线条更简单等）：")
                if st.button("🔄 根据指令重新生成掩码图"):
                    with st.spinner("AI 正在根据您的指令重新作画..."):
                        combined_text = f"原始文本内容：{text_area}\n\n【用户特别修改指令，请务必优先满足】：{modify_prompt}"
                        new_img = generate_mask_image_by_ai(combined_text)
                        st.session_state["ai_cache_image"] = new_img
                        st.rerun()

        elif mask_mode == "AI匹配本地模版" and st.session_state.get("ai_cache_text") == text_area and st.session_state.get("ai_cache_mode") == mask_mode and st.session_state.get("ai_cache_image") is not None:
            st.success(f"当前缓存的 AI 匹配掩码文件：{st.session_state.get('ai_cache_filename')}（随意调整上方参数，该图不会消失）")
            st.image(st.session_state["ai_cache_image"], width=200)
