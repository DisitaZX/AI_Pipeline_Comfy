import textwrap
from PIL import Image, ImageDraw, ImageFont

def create_cover(
    input_image_path,
    text,
    output_path,
    font_path="C:\\Users\\Loopy\\Desktop\\comfyui\\ComfyUI\\output\\AI_VIDEO\\Montserrat-VariableFont_wght.ttf",
    font_size=80,
):
    img = Image.open(input_image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    width, height = img.size

    try:
        font = ImageFont.truetype(font_path, font_size)
    except:
        font = ImageFont.load_default()

    # Разбиваем текст на слова и каждое с новой строки
    words = text.split()
    wrapped_text = "\n".join(words)

    # Расчет центра
    bbox = draw.multiline_textbbox((0, 0), wrapped_text, font=font, align="center")
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    x = (width - text_width) / 2
    y = (height - text_height) / 2

    # Рисование
    draw.multiline_text(
        (x, y),
        wrapped_text,
        font=font,
        fill="white",
        align="center",
        stroke_width=8,
        stroke_fill="orange",
    )

    img.save(output_path)
    print(f"Обложка сохранена: {output_path}")

# Пример
"""create_cover(
    input_image_path="C:\\Users\\Loopy\\Desktop\\comfyui\\ComfyUI\\output\\image_gen\\1_c1_00001_.png",
    text="Супер папа: 1 часть",
    output_path="result_cover.jpg",
    font_size=150,
)"""