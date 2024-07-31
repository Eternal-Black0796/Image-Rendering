import glfw
from OpenGL.GL import *
from OpenGL.GL.shaders import compileShader, compileProgram
import numpy as np
from PIL import Image, ImageEnhance
import os
import cv2
import tkinter as tk
from tkinter import filedialog
from tqdm import tqdm

# 顶点着色器代码
vertex_shader_code = """
#version 330
in vec2 position;
in vec2 texCoord;
out vec2 fragTexCoord;
void main() {
    gl_Position = vec4(position, 0.0, 1.0);
    fragTexCoord = texCoord;
}
"""

# 片段着色器代码
fragment_shader_code = """
#version 330
in vec2 fragTexCoord;
out vec4 color;
uniform sampler2D texture1;
void main() {
    vec4 texColor = texture(texture1, fragTexCoord);
    color = texColor;
}
"""

def apply_ray_tracing(img_array):
    kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])  # 边缘检测
    img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    img_edge = cv2.filter2D(img_gray, -1, kernel)
    img_edge_colored = cv2.cvtColor(img_edge, cv2.COLOR_GRAY2RGB)
    img_array = cv2.addWeighted(img_array, 0.8, img_edge_colored, 0.2, 0)
    return img_array

def enhance_image(image_path, output_path):
    img = Image.open(image_path).convert('RGB')
    img_array = np.array(img)
    img_array = apply_ray_tracing(img_array)
    img = Image.fromarray(img_array)
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.5)
    enhancer = ImageEnhance.Color(img)
    img = enhancer.enhance(1.5)
    img.save(output_path, format='PNG')  # 保存为PNG格式

def process_images(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    image_files = [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f)) and f.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif'))]
    for image_file in tqdm(image_files, desc="Processing images", unit="image"):
        input_path = os.path.join(input_folder, image_file)
        output_path = os.path.join(output_folder, os.path.splitext(image_file)[0] + '.png')  # 确保输出为PNG格式
        enhance_image(input_path, output_path)

def select_directory(prompt):
    root = tk.Tk()
    root.withdraw()
    directory = filedialog.askdirectory(title=prompt)
    return directory

def initialize_gl(width, height):
    if not glfw.init():
        raise Exception("GLFW could not be initialized")

    # Create a hidden window
    window = glfw.create_window(width, height, "Hidden Window", None, None)
    if not window:
        glfw.terminate()
        raise Exception("GLFW window could not be created")

    glfw.make_context_current(window)
    glfw.hide_window(window)  # Hide the window

    vertex_shader = compileShader(vertex_shader_code, GL_VERTEX_SHADER)
    fragment_shader = compileShader(fragment_shader_code, GL_FRAGMENT_SHADER)
    shader_program = compileProgram(vertex_shader, fragment_shader)
    glUseProgram(shader_program)

    vertices = np.array([
        -1.0, -1.0, 0.0, 0.0,  # Bottom-left
         1.0, -1.0, 1.0, 0.0,  # Bottom-right
         0.0,  1.0, 0.5, 1.0   # Top
    ], dtype=np.float32)

    vao = glGenVertexArrays(1)
    glBindVertexArray(vao)
    vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

    position = glGetAttribLocation(shader_program, "position")
    texCoord = glGetAttribLocation(shader_program, "texCoord")
    glVertexAttribPointer(position, 2, GL_FLOAT, False, 4 * vertices.itemsize, ctypes.c_void_p(0))
    glVertexAttribPointer(texCoord, 2, GL_FLOAT, False, 4 * vertices.itemsize, ctypes.c_void_p(2 * vertices.itemsize))
    glEnableVertexAttribArray(position)
    glEnableVertexAttribArray(texCoord)

    texture = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

    return vao, texture, shader_program, window

def render(vao, texture):
    glClear(GL_COLOR_BUFFER_BIT)
    glBindVertexArray(vao)
    glBindTexture(GL_TEXTURE_2D, texture)
    glDrawArrays(GL_TRIANGLES, 0, 3)

def main():
    width, height = 800, 600
    vao, texture, shader_program, window = initialize_gl(width, height)

    print("选择输入图片文件夹...")
    input_folder = select_directory("请选择输入图片文件夹")
    if not input_folder:
        print("没有选择输入文件夹，程序退出。")
        glfw.terminate()
        return

    print("选择输出图片文件夹...")
    output_folder = select_directory("请选择输出图片文件夹")
    if not output_folder:
        print("没有选择输出文件夹，程序退出。")
        glfw.terminate()
        return

    process_images(input_folder, output_folder)
    print("图片处理完成！")

    # 加载一张示例图像作为纹理
    example_image_path = os.path.join(output_folder, os.listdir(output_folder)[0])
    image = Image.open(example_image_path).convert('RGB')
    image_data = np.array(image)
    glBindTexture(GL_TEXTURE_2D, texture)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, image_data.shape[1], image_data.shape[0], 0, GL_RGB, GL_UNSIGNED_BYTE, image_data)

    # 渲染循环
    render(vao, texture)
    glfw.swap_buffers(window)

    print("按下回车键退出...")
    input()  # 等待用户按下回车键

    glfw.terminate()

if __name__ == "__main__":
    main()
