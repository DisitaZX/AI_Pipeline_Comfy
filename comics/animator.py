import os
from moviepy import ImageClip, ColorClip, CompositeVideoClip

class ImagePanner:
    def __init__(self, screen_width=720, screen_height=1280, bg_color=(0, 0, 0), max_speed=100):

        self.screen_size = (screen_width, screen_height)
        self.bg_color = bg_color
        self.move_downward = True
        self.max_speed = max_speed # Сохраняем лимит скорости

    def animate_image(self, image_path, audio_duration, output_path="output.mp4", fps=30):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Файл не найден: {image_path}")

        img_clip = ImageClip(image_path)
        screen_w, screen_h = self.screen_size

        # Масштабируем по ширине экрана
        img_clip = img_clip.resized(width=screen_w)

        # Проверяем высоту (увеличиваем короткие картинки)
        if img_clip.size[1] <= screen_h:
            img_clip = img_clip.resized(height=int(screen_h * 1.3))

        img_clip = img_clip.with_duration(audio_duration)
        img_w, img_h = img_clip.size
        x_pos = (screen_w - img_w) / 2

        # Максимально возможное расстояние для скролла
        max_scroll = img_h - screen_h
        if max_scroll < 0:
            max_scroll = 0

        # --- НОВЫЙ БЛОК: ОГРАНИЧЕНИЕ СКОРОСТИ ---
        actual_scroll = max_scroll
        # Если аудио больше 0 секунд, проверяем требуемую скорость
        if audio_duration > 0:
            current_speed = max_scroll / audio_duration
            if current_speed > self.max_speed:
                # Если слишком быстро, урезаем дистанцию скролла
                actual_scroll = self.max_speed * audio_duration

        # Логика скролла с учетом actual_scroll
        if self.move_downward:
            def calculate_position(t):
                progress = t / audio_duration
                y_pos = -(actual_scroll * progress)
                return (x_pos, y_pos)
        else:
            def calculate_position(t):
                progress = t / audio_duration
                # Начинаем с самого низа картинки и поднимаемся на разрешенную дистанцию
                y_pos = -max_scroll + (actual_scroll * progress)
                return (x_pos, y_pos)

        moving_img = img_clip.with_position(calculate_position)
        bg_clip = ColorClip(size=self.screen_size, color=self.bg_color).with_duration(audio_duration)
        final_clip = CompositeVideoClip([bg_clip, moving_img])

        print(f"Генерация: {output_path} | Время: {audio_duration}с | Дистанция: {actual_scroll:.0f}px (из {max_scroll})")
        final_clip.write_videofile(output_path, fps=fps, codec="libx264", audio=False)

        self.move_downward = not self.move_downward


# --- Пример использования ---
if __name__ == "__main__":
    panner = ImagePanner()

    # Не забудьте указать путь к существующей картинке
    # panner.animate_image("your_image.jpg", audio_duration=5.0, output_path="video_1.mp4")