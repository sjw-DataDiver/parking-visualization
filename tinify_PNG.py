import tinify
import os

# 配置你的API密钥  https://tinypng.com/developers
tinify.key = " XXX"

# 定义需要压缩的图片目录（可视化的frames目录）
img_dirs = ["./web_complete/frames/rgb_front", "./web_complete/frames/rgb_left",
            "./web_complete/frames/rgb_right", "./web_complete/frames/rgb_rear"]

# 批量压缩所有PNG
for dir_path in img_dirs:
    if not os.path.exists(dir_path):
        continue
    for file_name in os.listdir(dir_path):
        if file_name.endswith(".png"):
            file_path = os.path.join(dir_path, file_name)
            # 压缩并覆盖原文件
            source = tinify.from_file(file_path)
            source.to_file(file_path)
            print(f"压缩完成：{file_path}")
print("所有PNG图片压缩完成！")
