from PIL import Image, ImageDraw, ImageFont
import textwrap

def create_cover(input_image_path, text, output_path, font_path="C:\\Users\\Loopy\\Desktop\\ComfyUI_windows_portable\\ComfyUI\\output\\AI_VIDEO\\Montserrat-VariableFont_wght.ttf", font_size=80):
    img = Image.open(input_image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    width, height = img.size

    try:
        font = ImageFont.truetype(font_path, font_size)
    except:
        font = ImageFont.load_default()

    # --- 1. АВТОПЕРЕНОС ТЕКСТА ---
    # Определяем, сколько символов примерно поместится в одну строку
    # Это примерный расчет: ширина картинки деленная на (размер шрифта * 0.5)
    avg_char_width = font_size * 0.6
    chars_per_line = int(width / avg_char_width)

    # Разбиваем текст на строки
    lines = textwrap.wrap(text, width=chars_per_line)
    wrapped_text = "\n".join(lines)

    # --- 2. РАСЧЕТ ЦЕНТРА ДЛЯ МНОГОСТРОЧНОГО ТЕКСТА ---
    # Получаем границы всего блока текста
    bbox = draw.multiline_textbbox((0, 0), wrapped_text, font=font, align="center")
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    x = (width - text_width) / 2
    y = (height - text_height) / 2

    # --- 3. РИСОВАНИЕ ---
    draw.multiline_text(
        (x, y),
        wrapped_text,
        font=font,
        fill="white",
        align="center",  # Центрирует строки относительно друг друга
        stroke_width=4,
        stroke_fill="orange"
    )

    img.save(output_path)
    print(f"Обложка сохранена: {output_path}")

# Пример использования:
create_cover(
    input_image_path=f"C:\\Users\\Loopy\\Desktop\\ComfyUI_windows_portable\\ComfyUI\\output\\image_gen\\1_00001_.png",
    text="Арка злодея: часть 3",
    output_path="result_cover.jpg",
    font_size=50
)