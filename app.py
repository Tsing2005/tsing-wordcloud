import streamlit as st
import jieba
from wordcloud import WordCloud
from PIL import Image, ImageOps
import numpy as np
from collections import Counter
import io
import os
import platform

st.set_page_config(page_title="全格式词云生成器", layout="wide")
st.title("☁️ 专业词云生成系统 (精准形状识别版)")

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
    """
    全新升级的图像识别算法：
    1. 优先处理带透明通道的图片 (WebP/PNG)
    2. 对于白底实物（如浅黄色星星），采用 RGB 通道分离识别法，彻底解决内部空洞问题！
    """
    # 1. 优先处理透明背景
    if image.mode in ('RGBA', 'LA') or (image.mode == 'P' and 'transparency' in image.info):
        image = image.convert("RGBA")
        r, g, b, a = image.split()
        if a.getextrema()[0] < 255: # 判断确实存在透明部分
            # Alpha大于10视为形状(0)，透明视为背景(255)
            mask = np.where(np.array(a) > 10, 0, 255).astype(np.uint8)
            return np.stack([mask]*3, axis=-1)

    # 2. 对付纯色底/白底实物图（解决浅黄色星星中间空洞问题）
    # 将图像转为 RGB 数组
    img_array = np.array(image.convert("RGB"))
    
    # 关键修改点：提取每个像素点 R, G, B 三个颜色中最暗的那一个值
    # 纯白背景的 min 值接近 255；而浅黄色的蓝色通道很低，min 值会远小于 255。
    min_channels = np.min(img_array, axis=2)
    
    # 如果最暗的通道仍然大于阈值（默认240），说明它就是白色背景 -> 255
    # 否则，只要沾一点别的颜色，它就是形状主体 -> 0
    mask = np.where(min_channels >= threshold, 255, 0).astype(np.uint8)
    
    return np.stack([mask]*3, axis=-1)

# ================= 2. 侧边栏：参数与模版选择 =================
st.sidebar.header("🎨 设计面板")

font_p = st.sidebar.text_input("字体路径", value=get_default_font())

st.sidebar.subheader("色彩方案")
bg_color = st.sidebar.color_picker("1. 画布外部背景", "#FFFFFF")
shape_inner_color = st.sidebar.color_picker("2. 形状内部底色", "#FDF5E6") # 默认换成适合星星的米黄色
text_colormap = st.sidebar.selectbox("3. 文字颜色主题", ["viridis", "Set1", "plasma", "Dark2", "spring", "magma"])

st.sidebar.subheader("图形模版识别")
# 新增：让用户可以微调抠图算法的敏感度
mask_threshold = st.sidebar.slider("抠图白底容差 (若形状中间有空洞，请调大此值)", min_value=150, max_value=255, value=245, step=5)

mask_source = st.sidebar.radio("选择模版来源：", ["本地 masks 文件夹", "上传自定义照片"])

mask_image_to_use = None
supported_formats = ('.png', '.jpg', '.jpeg', '.webp')

if mask_source == "本地 masks 文件夹":
    mask_dir = "masks"
    if os.path.exists(mask_dir):
        files = [f for f in os.listdir(mask_dir) if f.lower().endswith(supported_formats)]
        if files:
            selected_file = st.sidebar.selectbox("请选择一个默认图形：", files)
            mask_image_to_use = Image.open(os.path.join(mask_dir, selected_file))
            st.sidebar.image(mask_image_to_use, caption="当前图形预览", width=120)
        else:
            st.sidebar.warning(f"目录 '{mask_dir}' 下没有找到图片。")
    else:
        st.sidebar.error(f"未找到 '{mask_dir}' 文件夹。")
else:
    uploaded = st.sidebar.file_uploader("点击上传图片 (支持 WebP, PNG, JPG)", type=["png", "jpg", "jpeg", "webp"])
    if uploaded:
        mask_image_to_use = Image.open(uploaded)
        st.sidebar.image(mask_image_to_use, caption="上传成功", width=120)

# ================= 3. 主界面逻辑 =================
text_area = st.text_area("输入文字内容：", height=150, value="星星 图像识别 算法 修复 RGB 抠图 颜色距离 词云 填充 完美 自动化 WebP 解决 内部空洞 智能 渲染")

if text_area:
    seg_list = jieba.lcut(text_area)
    counts = dict(Counter([w for w in seg_list if len(w) > 1 and w.strip()]))
    
    if not counts:
        st.info("等待输入文本...")
        st.stop()

    words_to_show = st.multiselect("筛选词汇：", options=list(counts.keys()), default=list(counts.keys())[:50])

    if st.button("🚀 生成高清词云"):
        if not words_to_show:
            st.error("词汇列表为空")
        else:
            try:
                with st.spinner("正在执行高精度抠图与图层合成..."):
                    filtered_data = {w: counts[w] for w in words_to_show}
                    
                    mask_array = None
                    if mask_image_to_use:
                        # 传入滑动条设定的阈值
                        mask_array = process_mask_to_array(mask_image_to_use, threshold=mask_threshold)
                    
                    # 生成透明文字层
                    wc = WordCloud(
                        font_path=font_p,
                        background_color=None,
                        mode="RGBA",
                        mask=mask_array,
                        colormap=text_colormap,
                        width=1000 if mask_array is None else mask_array.shape[1],
                        height=1000 if mask_array is None else mask_array.shape[0],
                        contour_width=0
                    )
                    wc.generate_from_frequencies(filtered_data)
                    wc_layer = wc.to_image()
                    w, h = wc_layer.size

                    # 合成多图层
                    final_canvas = Image.new("RGBA", (w, h), bg_color)
                    
                    if mask_array is not None:
                        inner_layer = Image.new("RGBA", (w, h), shape_inner_color)
                        # 0是形状，255是背景，这里提取并反转
                        shape_mask_data = 255 - mask_array[:, :, 0]
                        shape_mask_img = Image.fromarray(shape_mask_data, mode="L")
                        final_canvas = Image.composite(inner_layer, final_canvas, shape_mask_img)

                    # 覆盖文字
                    final_result = Image.alpha_composite(final_canvas, wc_layer)

                    # 展示
                    st.image(final_result, caption="渲染结果", use_container_width=True)

                    
                    st.markdown("---")
                    st.subheader("📥 下载中心")
                    c1, c2 = st.columns(2)
                    
                    # PNG 导出
                    png_buf = io.BytesIO()
                    final_result.save(png_buf, format="PNG")
                    c1.download_button("🖼️ 下载 PNG", data=png_buf.getvalue(), file_name="cloud_star.png")
                    
                    # PDF 导出
                    pdf_buf = io.BytesIO()
                    final_result.convert("RGB").save(pdf_buf, format="PDF", resolution=300)
                    c2.download_button("📩 下载 PDF", data=pdf_buf.getvalue(), file_name="cloud_star.pdf")

            except Exception as e:
                st.error(f"渲染失败: {e}")
